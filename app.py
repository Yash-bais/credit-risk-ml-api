from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD MODEL ARTIFACTS AT STARTUP
# ============================================
print("Loading model artifacts...")

# load raw model (for backwards compatibility) and the full pipeline (preferred)
rf_model = None
pipeline = None
model_columns = None
scaler = None

try:
    rf_model = joblib.load('credit_risk_model.pkl')
    print('✓ Raw RF model loaded')
except FileNotFoundError:
    print('⚠️ Raw RF model not found (credit_risk_model.pkl)')

try:
    pipeline = joblib.load('credit_risk_model_pipeline.pkl')
    print('✓ Full pipeline loaded (preferred for inference)')
except FileNotFoundError:
    print('⚠️ Pipeline artifact not found (credit_risk_model_pipeline.pkl)')

try:
    model_columns = joblib.load('credit_risk_model_columns.pkl')
    print(f'✓ Model expects {len(model_columns)} features')
except FileNotFoundError:
    print('⚠️ Model columns file not found (credit_risk_model_columns.pkl)')

try:
    scaler = joblib.load('credit_risk_model_scaler.pkl')
    print('✓ Scaler loaded')
except FileNotFoundError:
    print('⚠️ Scaler artifact not found (credit_risk_model_scaler.pkl)')

# ============================================
# DIAGNOSTIC TEST AT STARTUP
# ============================================
def map_to_raw_columns(raw_cols, payload_row: dict):
    """Return a single-row DataFrame aligned with raw_cols using payload_row values.
    The function tries common alternative keys (person_*, loan_*, etc.) when needed.
    """
    mapping_alternatives = {
        'age': ['person_age', 'age'],
        'ed': ['ed', 'person_education', 'education'],
        'employ': ['employ', 'person_emp_length'],
        'address': ['address'],
        'income': ['income', 'person_income', 'annualIncome'],
        'debtinc': ['debtinc', 'loan_percent_income'],
        'creddebt': ['creddebt', 'cb_person_cred_hist_length'],
        'othdebt': ['othdebt', 'loan_amnt']
    }

    row = {}
    for col in raw_cols:
        val = None
        # try direct key
        if col in payload_row:
            val = payload_row.get(col)
        # try mapped alternatives
        elif col in mapping_alternatives:
            for alt in mapping_alternatives[col]:
                if alt in payload_row:
                    val = payload_row.get(alt)
                    break

        # final fallback — numeric zero or empty string
        if val is None:
            # guess numeric vs categorical by column name heuristics
            if isinstance(val, str) and val.strip() == '':
                val = 'MISSING'
            else:
                val = 0

        row[col] = val

    return pd.DataFrame([row])


if pipeline is not None:
    print("\n" + "="*70)
    print("STARTUP DIAGNOSTIC TEST")
    print("="*70)
    
    test_cases = [
        {
            'label': 'BAD (should be REJECTED)',
            'data': pd.DataFrame([{
                'person_age': 25,
                'person_income': 5000,
                'person_emp_length': 1,
                'loan_amnt': 150000,
                'loan_int_rate': 20.0,
                'loan_percent_income': 30.0,
                'cb_person_cred_hist_length': 1,
                'person_home_ownership': 'RENT',
                'loan_intent': 'PERSONAL',
                'loan_grade': 'G',
                'cb_person_default_on_file': 'Y'
            }])
        },
        {
            'label': 'GOOD (should be APPROVED)',
            'data': pd.DataFrame([{
                'person_age': 35,
                'person_income': 100000,
                'person_emp_length': 10,
                'loan_amnt': 10000,
                'loan_int_rate': 5.0,
                'loan_percent_income': 0.1,
                'cb_person_cred_hist_length': 15,
                'person_home_ownership': 'OWN',
                'loan_intent': 'PERSONAL',
                'loan_grade': 'A',
                'cb_person_default_on_file': 'N'
            }])
        }
    ]
    
    # helper to map person_* payload keys to the training raw column names
    def map_to_raw_columns(raw_cols, payload_row: dict):
        """Return a single-row DataFrame aligned with raw_cols using payload_row values.
        The function tries common alternative keys (person_*, loan_*, etc.) when needed.
        """
        mapping_alternatives = {
            'age': ['person_age', 'age'],
            'ed': ['ed', 'person_education', 'education'],
            'employ': ['employ', 'person_emp_length'],
            'address': ['address'],
            'income': ['income', 'person_income', 'annualIncome'],
            'debtinc': ['debtinc', 'loan_percent_income'],
            'creddebt': ['creddebt', 'cb_person_cred_hist_length'],
            'othdebt': ['othdebt', 'loan_amnt']
        }

        row = {}
        for col in raw_cols:
            val = None
            # try direct key
            if col in payload_row:
                val = payload_row.get(col)
            # try mapped alternatives
            elif col in mapping_alternatives:
                for alt in mapping_alternatives[col]:
                    if alt in payload_row:
                        val = payload_row.get(alt)
                        break

            # final fallback — numeric zero or empty string
            if val is None:
                # guess numeric vs categorical by column name heuristics
                if isinstance(val, str) and val.strip() == '':
                    val = 'MISSING'
                else:
                    val = 0

            row[col] = val

        return pd.DataFrame([row])

    # get expected raw columns from the ColumnTransformer inside the pipeline
    try:
        preproc = pipeline.named_steps['preprocessor']
        expected_raw_cols = []
        for _name, _trans, cols in preproc.transformers:
            # cols can be slice, list or array-like
            if isinstance(cols, (list, tuple)):
                expected_raw_cols.extend(list(cols))
        expected_raw_cols = list(dict.fromkeys(expected_raw_cols))
    except Exception:
        expected_raw_cols = []

    for test_case in test_cases:
        label = test_case['label']
        test_df = test_case['data']
        # we may receive a DataFrame with person_* columns — map to raw columns expected by pipeline
        if expected_raw_cols:
            try:
                mapped = map_to_raw_columns(expected_raw_cols, test_df.iloc[0].to_dict())
                X_test = pipeline.named_steps['preprocessor'].transform(mapped)
                pred = pipeline.predict(mapped)[0]
                proba = pipeline.predict_proba(mapped)[0]
            except Exception as e:
                print('Startup diagnostic mapping failed:', e)
                # as a fallback, try direct predict on incoming DataFrame
                pred = pipeline.predict(test_df)[0]
                proba = pipeline.predict_proba(test_df)[0]
                X_test = None
        else:
            # no expected_raw_cols discovered — try direct prediction
            pred = pipeline.predict(test_df)[0]
            proba = pipeline.predict_proba(test_df)[0]
            X_test = None
        
        print(f"\n{label}:")
        print(f"  Raw prediction: {pred}")
        print(f"  Probabilities: Class 0={proba[0]:.4f}, Class 1={proba[1]:.4f}")
        print(f"  Features sum: {X_test.sum():.2f}")
    
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("  ✓ If sums are DIFFERENT and NON-ZERO → Model working!")
    print("  ✗ If both sums are 0.00 → Feature encoding issue")
    print("="*70 + "\n")

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Credit Risk Prediction Endpoint
    """
    # require a pipeline (preferred) or at minimum a raw model
    if pipeline is None and rf_model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure all .pkl files exist.'
        }), 500
    
    try:
        data = request.json
        
        print(f"\n{'='*70}")
        print(f"NEW PREDICTION REQUEST at {datetime.now().isoformat()}")
        print(f"{'='*70}")
        
        # Extract and validate inputs
        try:
            person_age = int(data.get('person_age', 18))
            person_income = float(data.get('person_income', 0))
            person_emp_length = int(data.get('person_emp_length', 0))
            loan_amnt = float(data.get('loan_amnt', 0))
            loan_int_rate = float(data.get('loan_int_rate', 10.0))
            loan_percent_income = float(data.get('loan_percent_income', 0))
            cb_person_cred_hist_length = int(data.get('cb_person_cred_hist_length', 0))
            person_home_ownership = data.get('person_home_ownership', 'RENT').upper()
            loan_intent = data.get('loan_intent', 'PERSONAL').upper()
            loan_grade = data.get('loan_grade', 'C').upper()
            cb_person_default_on_file = data.get('cb_person_default_on_file', 'N').upper()
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
        
        # Validate
        if person_income <= 0:
            return jsonify({'error': 'Income must be greater than 0'}), 400
        if loan_amnt <= 0:
            return jsonify({'error': 'Loan amount must be greater than 0'}), 400
        
        # Calculate loan_percent_income if needed
        if loan_percent_income == 0 and person_income > 0:
            loan_percent_income = (loan_amnt / person_income) * 100
        
        print(f"Input: Age={person_age}, Income=${person_income}, Loan=${loan_amnt}")
        print(f"       Grade={loan_grade}, Defaults={cb_person_default_on_file}")
        
        # Create input DataFrame with exact column order as training
        input_df = pd.DataFrame([{
            'person_age': person_age,
            'person_income': person_income,
            'person_emp_length': person_emp_length,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'person_home_ownership': person_home_ownership,
            'loan_intent': loan_intent,
            'loan_grade': loan_grade,
            'cb_person_default_on_file': cb_person_default_on_file
        }])
        
        # If a full sklearn Pipeline is available prefer that (it does preprocessing + classifier)
        if pipeline is not None:
            # get expected raw columns used by the pipeline preprocessor (if possible)
            try:
                preproc = pipeline.named_steps['preprocessor']
                expected_raw_cols = []
                for _name, _trans, cols in preproc.transformers:
                    if isinstance(cols, (list, tuple)):
                        expected_raw_cols.extend(list(cols))
                expected_raw_cols = list(dict.fromkeys(expected_raw_cols))
            except Exception:
                expected_raw_cols = []

            # if the pipeline expects raw columns (age, ed, ...) map input to them first
            if expected_raw_cols:
                mapped_input = map_to_raw_columns(expected_raw_cols, data)
            else:
                # fallback use the person_* DataFrame we've already created
                mapped_input = input_df

            prediction_numeric = pipeline.predict(mapped_input)[0]
            prediction_proba = pipeline.predict_proba(mapped_input)[0]
            # also get transformed features for diagnostics (if possible)
            try:
                X_transformed = pipeline.named_steps['preprocessor'].transform(mapped_input)
                print(f"Transformed shape: {X_transformed.shape}")
                print(f"Features sum: {X_transformed.sum():.2f}")
            except Exception:
                X_transformed = None
        else:
            # fallback: attempt to use raw rf_model (best-effort). This requires the model to accept transformed input
            X_transformed = None
            # If we only have a raw model, try to produce a feature vector that matches model_columns
            if model_columns:
                # model_columns are the post-encoding feature names saved at training time
                # best-effort: try to map incoming fields to original raw columns and then
                # rely on scaler + ordering if present
                try:
                    # if model_columns appear to be plain raw column names (no one-hot), map directly
                    mapped = map_to_raw_columns(model_columns, data)
                    # if scaler is available, transform numeric portion
                    if scaler is not None:
                        try:
                            X_for_model = scaler.transform(mapped[model_columns])
                        except Exception:
                            X_for_model = mapped.values
                    else:
                        X_for_model = mapped.values
                except Exception:
                    X_for_model = input_df.values

                prediction_numeric = rf_model.predict(X_for_model)[0]
                prediction_proba = rf_model.predict_proba(X_for_model)[0]
            else:
                # fall back to raw input DataFrame
                prediction_numeric = rf_model.predict(input_df)[0]
                prediction_proba = rf_model.predict_proba(input_df)[0]
        
        print(f"Raw prediction: {prediction_numeric}")
        print(f"Probabilities: Class 0={prediction_proba[0]:.4f}, Class 1={prediction_proba[1]:.4f}")
        
        # Map prediction: 0 = no default (approved), 1 = default (rejected)
        prediction = "rejected" if prediction_numeric == 1 else "approved"
        confidence = float(prediction_proba[prediction_numeric])
        
        print(f"Final decision: {prediction.upper()} with {confidence:.4f} confidence")
        print(f"{'='*70}\n")
        
        # Generate risk factors
        risk_factors = []
        
        # Age
        if person_age < 25:
            risk_factors.append({
                "factor": "age",
                "impact": "negative",
                "description": f"Young age ({person_age}) - limited financial history"
            })
        elif person_age > 65:
            risk_factors.append({
                "factor": "age",
                "impact": "negative",
                "description": f"Advanced age ({person_age}) - retirement concerns"
            })
        
        # Income
        if person_income < 30000:
            risk_factors.append({
                "factor": "income",
                "impact": "negative",
                "description": f"Low annual income (${person_income:,.0f})"
            })
        elif person_income > 100000:
            risk_factors.append({
                "factor": "income",
                "impact": "positive",
                "description": f"High annual income (${person_income:,.0f})"
            })
        
        # Loan-to-income ratio
        if loan_percent_income > 40:
            risk_factors.append({
                "factor": "loan_to_income",
                "impact": "negative",
                "description": f"High loan-to-income ratio ({loan_percent_income:.1f}%)"
            })
        elif loan_percent_income < 20:
            risk_factors.append({
                "factor": "loan_to_income",
                "impact": "positive",
                "description": f"Low loan-to-income ratio ({loan_percent_income:.1f}%)"
            })
        
        # Employment
        if person_emp_length < 2:
            risk_factors.append({
                "factor": "employment",
                "impact": "negative",
                "description": f"Short employment history ({person_emp_length} years)"
            })
        elif person_emp_length >= 5:
            risk_factors.append({
                "factor": "employment",
                "impact": "positive",
                "description": f"Stable employment ({person_emp_length} years)"
            })
        
        # Credit history
        if cb_person_cred_hist_length < 3:
            risk_factors.append({
                "factor": "credit_history",
                "impact": "negative",
                "description": f"Limited credit history ({cb_person_cred_hist_length} years)"
            })
        elif cb_person_cred_hist_length >= 10:
            risk_factors.append({
                "factor": "credit_history",
                "impact": "positive",
                "description": f"Extensive credit history ({cb_person_cred_hist_length} years)"
            })
        
        # Previous defaults
        if cb_person_default_on_file == 'Y':
            risk_factors.append({
                "factor": "default_history",
                "impact": "negative",
                "description": "Previous default on file - major red flag"
            })
        else:
            risk_factors.append({
                "factor": "default_history",
                "impact": "positive",
                "description": "No previous defaults - good payment history"
            })
        
        # Interest rate
        if loan_int_rate > 15:
            risk_factors.append({
                "factor": "interest_rate",
                "impact": "negative",
                "description": f"High interest rate ({loan_int_rate}%)"
            })
        elif loan_int_rate < 8:
            risk_factors.append({
                "factor": "interest_rate",
                "impact": "positive",
                "description": f"Low interest rate ({loan_int_rate}%)"
            })
        
        # Loan grade
        if loan_grade in ['D', 'E', 'F', 'G']:
            risk_factors.append({
                "factor": "loan_grade",
                "impact": "negative",
                "description": f"Low loan grade ({loan_grade})"
            })
        elif loan_grade in ['A', 'B']:
            risk_factors.append({
                "factor": "loan_grade",
                "impact": "positive",
                "description": f"Excellent loan grade ({loan_grade})"
            })
        
        # Home ownership
        if person_home_ownership == 'OWN':
            risk_factors.append({
                "factor": "home_ownership",
                "impact": "positive",
                "description": "Owns home - financial stability"
            })
        elif person_home_ownership == 'MORTGAGE':
            risk_factors.append({
                "factor": "home_ownership",
                "impact": "positive",
                "description": "Has mortgage - demonstrates creditworthiness"
            })
        
        # Loan amount
        if loan_amnt > 35000:
            risk_factors.append({
                "factor": "loan_amount",
                "impact": "negative",
                "description": f"Large loan amount (${loan_amnt:,.0f})"
            })
        
        # Return response
        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'risk_factors': risk_factors
        })
        
    except Exception as e:
        error_msg = f'Prediction error: {str(e)}'
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if rf_model else 'not_loaded',
        'timestamp': datetime.now().isoformat(),
        'service': 'Credit Risk Prediction API'
    })

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'service': 'Credit Risk Prediction API',
        'version': '3.0.0',
        'model': 'Random Forest with Pipeline',
        'model_status': 'loaded' if rf_model else 'not_loaded',
        'features_count': len(model_columns) if model_columns else 0,
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        }
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("Credit Risk Prediction API")
    print("="*70)
    if rf_model is not None:
        print("✓ Model ready for predictions")
    else:
        print("✗ Model NOT loaded")
    print("="*70)
    print("Starting server on http://0.0.0.0:5000")
    print("="*70 + "\n")
    app.run(debug=True, port=5000, host='0.0.0.0')