from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import os

CLEANUP_TRAINING = False

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================
# LOAD MODEL ARTIFACTS AT STARTUP
# ============================================
def train_model(data_path: str, output_prefix: str = 'credit_risk_model'):
    """Train a RandomForest model from a training CSV and save artifacts.

    This mirrors the existing standalone training script. If you want to use
    the dataset from disk, put the CSV in the project root (next to `app.py`)
    or pass --data with a path.
    """
    import pandas as pd
    import joblib
    import os

    if not os.path.exists(data_path):
        print(f"ERROR: training data not found at: {data_path}")
        return False

    df = pd.read_csv(data_path)
    print(f"Loaded training CSV: {data_path} (shape={df.shape})")

    # --------------- detect target column ------------------
    possible_targets = ['loan_status', 'default', 'target', 'loan_default']
    target_col = None
    for t in possible_targets:
        if t in df.columns:
            target_col = t
            break

    if target_col is None:
        print(f"ERROR: No target column found. expected one of {possible_targets}")
        return False

    # --------------- detect feature schema ------------------
    # We support two common schemas: the 'bankloans' minimal numeric features
    # (age, ed, employ, address, income, debtinc, creddebt, othdebt) or the
    # API-style person_* columns.
    bank_features = ['age', 'ed', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt']
    person_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

    if all(col in df.columns for col in bank_features):
        # Drop rows missing the target
        df = df.dropna(subset=[target_col])
        X = df[bank_features + [c for c in df.columns if c not in bank_features + [target_col]]]
        y = df[target_col]
        print('Detected bankloans numeric schema; using numeric features')
    elif any(col in df.columns for col in person_features):
        # fallback: use all columns except target
        df = df.dropna(subset=[target_col])
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        print('Detected API/person_* style schema; using all columns except the target')
    else:
        # last resort: use all columns except target
        df = df.dropna(subset=[target_col])
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        print('Using all columns except the target as features')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # train RandomForest
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)

    # evaluate quickly
    y_pred = rf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Training done. Test accuracy: {acc:.4f}")

    # Save artifacts - keep names existing code expects for compatibility
    joblib.dump(rf, f"{output_prefix}.pkl")
    joblib.dump(scaler, f"{output_prefix}_scaler.pkl")
    model_columns = list(X.columns)
    joblib.dump(model_columns, f"{output_prefix}_columns.pkl")

    # Save a combined pipeline too (recommended for serving)
    pipeline = Pipeline([('scaler', scaler), ('classifier', rf)])
    joblib.dump(pipeline, f"{output_prefix}_pipeline.pkl")

    print('Saved artifacts: ', [f"{output_prefix}.pkl", f"{output_prefix}_scaler.pkl", f"{output_prefix}_columns.pkl", f"{output_prefix}_pipeline.pkl"]) 
    return True

rf_model = None
scaler = None
model_columns = None
pipeline = None

def load_artifacts():
    """Load model artifacts into the global names rf_model, scaler, model_columns.

    When available this prefers the saved pipeline artifact for consistency.
    """
    global rf_model, scaler, model_columns

    print("Loading model artifacts...")
    try:
        # Prefer to load the combined pipeline if present
        if os.path.exists('credit_risk_model_pipeline.pkl'):
            loaded_pipeline = joblib.load('credit_risk_model_pipeline.pkl')
            globals()['pipeline'] = loaded_pipeline
            rf_model = loaded_pipeline.named_steps.get('classifier')
            scaler = loaded_pipeline.named_steps.get('scaler')
        else:
            rf_model = joblib.load('credit_risk_model.pkl')
            scaler = joblib.load('credit_risk_model_scaler.pkl') if os.path.exists('credit_risk_model_scaler.pkl') else joblib.load('scaler.pkl')

        model_columns = joblib.load('credit_risk_model_columns.pkl') if os.path.exists('credit_risk_model_columns.pkl') else joblib.load('model_columns.pkl')
        print(f"✓ Model loaded successfully")
        print(f"✓ Scaler loaded successfully")
        print(f"✓ Model expects {len(model_columns)} features")
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found. Please run the training script first.")
        print(f"Missing file: {e}")
        rf_model = None
        scaler = None
        model_columns = None

def run_startup_diagnostic():
    if rf_model is None or model_columns is None:
        print('Skipping startup diagnostics because model is not loaded.')
        return
    print("\n" + "="*70)
    print("TRAINING FEATURES (what the model expects):")
    print("="*70)
    for i, col in enumerate(model_columns):
        print(f"  {i+1:2d}. {col}")
    print("="*70)
    print(f"Total training features: {len(model_columns)}")
    print("="*70 + "\n")
    
    print("="*70)
    print("TESTING MODEL WITH TWO APPLICATIONS")
    print("="*70)
    
    test_cases = [
        {
            'label': 'BAD (should be REJECTED)',
            'data': {
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
            }
        },
        {
            'label': 'GOOD (should be APPROVED)',
            'data': {
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
            }
        }
    ]
    
    for test_case in test_cases:
        label = test_case['label']
        test_data = test_case['data']
        
        print(f"\n{label}:")
        print(f"  Input: Income=${test_data['person_income']}, Loan=${test_data['loan_amnt']}, Grade={test_data['loan_grade']}, Defaults={test_data['cb_person_default_on_file']}")
        
        # Create DataFrame
        df_test = pd.DataFrame([test_data])
        
        # One-hot encode
        df_test_encoded = pd.get_dummies(
            df_test, 
            columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'],
            drop_first=True
        )
        
        # Align with training features
        for col in model_columns:
            if col not in df_test_encoded.columns:
                df_test_encoded[col] = 0
        
        # Keep only training columns in correct order
        df_test_aligned = df_test_encoded[model_columns]
        
        # Scale
        X_test_scaled = scaler.transform(df_test_aligned)
        
        # Predict
        pred = rf_model.predict(X_test_scaled)[0]
        proba = rf_model.predict_proba(X_test_scaled)[0]
        
        print(f"  Raw prediction: {pred}")
        print(f"  Probabilities: Class 0={proba[0]:.4f}, Class 1={proba[1]:.4f}")
        print(f"  Features sum: {df_test_aligned.values.sum():.2f} (should be different for each test)")
        
    print("="*70)
    print("DIAGNOSIS:")
    print("  - If BOTH tests have IDENTICAL probabilities -> Feature mismatch issue")
    print("  - If probabilities are DIFFERENT -> Model is working, check label mapping")
    print("="*70 + "\n")

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Credit Risk Prediction Endpoint
    Receives loan application data and returns prediction with confidence and risk factors
    """
    
    # Check if model is loaded
    if rf_model is None:
        return jsonify({
            'error': 'Model not loaded. Please ensure credit_risk_model.pkl, scaler.pkl, and model_columns.pkl exist.'
        }), 500
    
    try:
        data = request.json
        
        # Log incoming request
        print(f"\n{'='*70}")
        print(f"NEW PREDICTION REQUEST at {datetime.now().isoformat()}")
        print(f"{'='*70}")

        # If we have a combined pipeline and the model's expected columns look
        # numeric (bankloans schema), try to map & predict using the pipeline
        if globals().get('pipeline', None) is not None and model_columns is not None:
            expected_cols = list(model_columns)

            # quick check for numeric-schema signature
            numeric_signature = set(['age', 'income', 'employ', 'ed', 'address', 'debtinc', 'creddebt', 'othdebt'])
            if numeric_signature.intersection(set(expected_cols)):
                # mapping helper (same as brick used later)
                def build_features_for_pipeline(req_json, expected_cols):
                    mapping = {
                        'age': ['age', 'person_age'],
                        'ed': ['ed'],
                        'employ': ['employ', 'person_emp_length', 'employmentYears'],
                        'address': ['address'],
                        'income': ['income', 'person_income', 'annualIncome'],
                        'debtinc': ['debtinc', 'loan_percent_income', 'loanPercentIncome'],
                        'creddebt': ['creddebt', 'credit_debt', 'creditScore', 'cb_person_cred_hist_length'],
                        'othdebt': ['othdebt', 'loan_amnt', 'loanAmount']
                    }

                    out = {}
                    for col in expected_cols:
                        val = None
                        if col in req_json and req_json.get(col) is not None:
                            val = req_json.get(col)
                        else:
                            for alt in mapping.get(col, []):
                                if alt in req_json and req_json.get(alt) is not None:
                                    val = req_json.get(alt)
                                    break

                        if val is None:
                            if col == 'debtinc':
                                loan = req_json.get('loan_amnt') or req_json.get('loanAmount')
                                inc = req_json.get('person_income') or req_json.get('income') or req_json.get('annualIncome')
                                if loan and inc:
                                    try:
                                        val = (float(loan) / float(inc)) * 100
                                    except Exception:
                                        val = 0.0
                                else:
                                    val = 0.0
                            elif col == 'creddebt':
                                cs = req_json.get('creditScore')
                                if cs is not None:
                                    try:
                                        val = float(cs) / 10.0
                                    except Exception:
                                        val = 0.0
                                else:
                                    val = 0.0
                            else:
                                val = 0.0

                        try:
                            val = float(val)
                        except Exception:
                            pass

                        out[col] = val

                    return pd.DataFrame([out], columns=expected_cols)

                feature_df = build_features_for_pipeline(data, expected_cols)
                print('Built pipeline input:', feature_df.iloc[0].to_dict())

                pipe = globals().get('pipeline')
                try:
                    if hasattr(pipe, 'predict_proba'):
                        proba = pipe.predict_proba(feature_df)[0]
                        pred_num = pipe.predict(feature_df)[0]
                    else:
                        scaled = scaler.transform(feature_df)
                        proba = rf_model.predict_proba(scaled)[0]
                        pred_num = rf_model.predict(scaled)[0]
                except Exception as e:
                    print('Pipeline prediction failed:', e)
                else:
                    prediction = 'rejected' if int(pred_num) == 1 else 'approved'
                    confidence = float(max(proba))
                    risk_factors = []
                    if feature_df['age'].iloc[0] < 25:
                        risk_factors.append({'factor': 'age', 'impact': 'negative', 'description': f"Young age ({feature_df['age'].iloc[0]})"})
                    if feature_df['income'].iloc[0] > 100000:
                        risk_factors.append({'factor': 'income', 'impact': 'positive', 'description': f"High income (${feature_df['income'].iloc[0]:,.0f})"})
                    return jsonify({'prediction': prediction, 'confidence': round(confidence,4), 'risk_factors': risk_factors})
        
        # ============================================
        # STEP 1: Extract and validate input data
        # ============================================
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
            return jsonify({'error': f'Invalid input data type: {str(e)}'}), 400
        
        # Validate ranges
        if person_income <= 0:
            return jsonify({'error': 'Annual income must be greater than 0'}), 400
        if loan_amnt <= 0:
            return jsonify({'error': 'Loan amount must be greater than 0'}), 400
        if person_age < 18 or person_age > 100:
            return jsonify({'error': 'Age must be between 18 and 100'}), 400
        
        # Calculate loan_percent_income if needed (with zero-check guard)
        if loan_percent_income == 0 and person_income > 0:
            loan_percent_income = (loan_amnt / person_income) * 100
        
        print(f"Input: Age={person_age}, Income=${person_income}, Loan=${loan_amnt}, Grade={loan_grade}, Defaults={cb_person_default_on_file}")
        
        # ============================================
        # STEP 2: Create base features dataframe
        # ============================================
        input_data = {
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
        }
        
        df_input = pd.DataFrame([input_data])

        # If we loaded a combined pipeline (preprocessor + model), prefer using it.
        # The pipeline we trained on bankloans.csv expects numeric features like
        # ['age','ed','employ','address','income','debtinc','creddebt','othdebt'].
        if globals().get('pipeline', None) is not None and model_columns is not None:
            expected_cols = list(model_columns)

            # helper that maps request JSON to expected numeric columns
            def build_features_for_pipeline(req_json, expected_cols):
                mapping = {
                    'age': ['age', 'person_age'],
                    'ed': ['ed'],
                    'employ': ['employ', 'person_emp_length', 'employmentYears'],
                    'address': ['address'],
                    'income': ['income', 'person_income', 'annualIncome'],
                    'debtinc': ['debtinc', 'loan_percent_income', 'loanPercentIncome'],
                    'creddebt': ['creddebt', 'credit_debt', 'creditScore', 'cb_person_cred_hist_length'],
                    'othdebt': ['othdebt', 'loan_amnt', 'loanAmount']
                }

                out = {}
                for col in expected_cols:
                    val = None
                    # Check direct key match first
                    if col in req_json and req_json.get(col) is not None:
                        val = req_json.get(col)
                    else:
                        for alt in mapping.get(col, []):
                            if alt in req_json and req_json.get(alt) is not None:
                                val = req_json.get(alt)
                                break

                    # Derived calculations
                    if val is None:
                        if col == 'debtinc':
                            loan = req_json.get('loan_amnt') or req_json.get('loanAmount')
                            inc = req_json.get('person_income') or req_json.get('income') or req_json.get('annualIncome')
                            if loan and inc:
                                try:
                                    val = (float(loan) / float(inc)) * 100
                                except Exception:
                                    val = 0.0
                            else:
                                val = 0.0
                        elif col == 'creddebt':
                            cs = req_json.get('creditScore')
                            if cs is not None:
                                try:
                                    val = float(cs) / 10.0
                                except Exception:
                                    val = 0.0
                            else:
                                val = 0.0
                        else:
                            val = 0.0

                    # convert to float if possible
                    try:
                        val = float(val)
                    except Exception:
                        pass

                    out[col] = val

                return pd.DataFrame([out], columns=expected_cols)

            # build feature row and run through pipeline
            feature_df = build_features_for_pipeline(data, expected_cols)
            print(f"Built pipeline input: {feature_df.iloc[0].to_dict()}")

            # run through pipeline
            pipe = globals().get('pipeline')
            X_pred = pipe.transform(feature_df) if hasattr(pipe, 'transform') and not hasattr(pipe, 'named_steps') else feature_df.values
            try:
                # If pipeline has predict or predict_proba methods, use them directly
                if hasattr(pipe, 'predict_proba'):
                    proba = pipe.predict_proba(feature_df)[0]
                    pred_num = pipe.predict(feature_df)[0]
                else:
                    # fallback: use rf_model + scaler
                    scaled = scaler.transform(feature_df)
                    proba = rf_model.predict_proba(scaled)[0]
                    pred_num = rf_model.predict(scaled)[0]
            except Exception as e:
                # as a safe fallback, continue with existing code path
                print('Pipeline prediction failed:', e)
            else:
                prediction = 'rejected' if int(pred_num) == 1 else 'approved'
                confidence = float(max(proba))
                risk_factors = []
                # minimal risk factors using numeric schema
                if feature_df['age'].iloc[0] < 25:
                    risk_factors.append({'factor': 'age', 'impact': 'negative', 'description': f"Young age ({feature_df['age'].iloc[0]})"})
                if feature_df['income'].iloc[0] > 100000:
                    risk_factors.append({'factor': 'income', 'impact': 'positive', 'description': f"High income (${feature_df['income'].iloc[0]:,.0f})"})
                # return early with pipeline result
                return jsonify({'prediction': prediction, 'confidence': round(confidence,4), 'risk_factors': risk_factors})
        
        print(f"DataFrame created with shape: {df_input.shape}")
        
        # ============================================
        # STEP 3: Apply one-hot encoding (same as training)
        # ============================================
        df_encoded = pd.get_dummies(
            df_input, 
            columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'],
            drop_first=True
        )
        
        print(f"After encoding: {df_encoded.shape[1]} features")
        print(f"Encoded columns: {list(df_encoded.columns)}")
        
        # ============================================
        # STEP 4: Align features with training data
        # ============================================
        # Add missing columns with 0 values
        missing_cols = []
        for col in model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                missing_cols.append(col)
        
        if missing_cols:
            print(f"Added {len(missing_cols)} missing columns (set to 0): {missing_cols[:5]}{'...' if len(missing_cols) > 5 else ''}")
        
        # Keep only the columns that were in training (in the same order)
        df_aligned = df_encoded[model_columns]
        
        print(f"After alignment: {df_aligned.shape}")
        print(f"Features match training? {list(df_aligned.columns) == model_columns}")
        print(f"Non-zero features: {(df_aligned != 0).sum().sum()}")
        print(f"Features sum: {df_aligned.values.sum():.2f}")
        
        # Debug: Print first 10 non-zero features
        non_zero_features = df_aligned.iloc[0][df_aligned.iloc[0] != 0]
        print(f"Non-zero feature values (first 10):")
        for i, (feat, val) in enumerate(non_zero_features.items()):
            if i >= 10:
                break
            print(f"  {feat}: {val}")
        
        # ============================================
        # STEP 5: Scale features
        # ============================================
        X_scaled = scaler.transform(df_aligned)
        print(f"Scaled features range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
        
        # ============================================
        # STEP 6: Make prediction
        # ============================================
        prediction_numeric = rf_model.predict(X_scaled)[0]
        prediction_proba = rf_model.predict_proba(X_scaled)[0]
        
        print(f"Raw prediction: {prediction_numeric}")
        print(f"Prediction probabilities: Class 0={prediction_proba[0]:.4f}, Class 1={prediction_proba[1]:.4f}")
        
        # Map prediction to approval/rejection
        # Based on typical credit datasets: 0 = no default (approve), 1 = default (reject)
        prediction = "rejected" if prediction_numeric == 1 else "approved"
        confidence = float(prediction_proba.max())
        
        print(f"Final decision: {prediction.upper()} with {confidence:.4f} confidence")
        print(f"{'='*70}\n")
        
        # ============================================
        # STEP 7: Generate risk factors explanation
        # ============================================
        risk_factors = []
        
        # Age risk
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
                "description": f"Advanced age ({person_age}) - retirement income concerns"
            })
        
        # Income analysis
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
        
        # Loan to income ratio
        if loan_percent_income > 40:
            risk_factors.append({
                "factor": "loan_to_income",
                "impact": "negative",
                "description": f"High loan-to-income ratio ({loan_percent_income:.1f}%) - potential repayment difficulty"
            })
        elif loan_percent_income < 20:
            risk_factors.append({
                "factor": "loan_to_income",
                "impact": "positive",
                "description": f"Low loan-to-income ratio ({loan_percent_income:.1f}%) - affordable repayment"
            })
        
        # Employment stability
        if person_emp_length < 2:
            risk_factors.append({
                "factor": "employment",
                "impact": "negative",
                "description": f"Short employment history ({person_emp_length} years) - income stability concern"
            })
        elif person_emp_length >= 5:
            risk_factors.append({
                "factor": "employment",
                "impact": "positive",
                "description": f"Stable employment history ({person_emp_length} years)"
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
                "description": "Previous default on credit file - major red flag"
            })
        else:
            risk_factors.append({
                "factor": "default_history",
                "impact": "positive",
                "description": "No previous defaults - good payment history"
            })
        
        # Interest rate analysis
        if loan_int_rate > 15:
            risk_factors.append({
                "factor": "interest_rate",
                "impact": "negative",
                "description": f"High interest rate ({loan_int_rate}%) - indicates higher risk assessment"
            })
        elif loan_int_rate < 8:
            risk_factors.append({
                "factor": "interest_rate",
                "impact": "positive",
                "description": f"Low interest rate ({loan_int_rate}%) - favorable terms"
            })
        
        # Loan grade
        if loan_grade in ['D', 'E', 'F', 'G']:
            risk_factors.append({
                "factor": "loan_grade",
                "impact": "negative",
                "description": f"Low loan grade ({loan_grade}) - higher default risk"
            })
        elif loan_grade in ['A', 'B']:
            risk_factors.append({
                "factor": "loan_grade",
                "impact": "positive",
                "description": f"Excellent loan grade ({loan_grade}) - low risk borrower"
            })
        
        # Home ownership
        if person_home_ownership == 'OWN':
            risk_factors.append({
                "factor": "home_ownership",
                "impact": "positive",
                "description": "Owns home - shows financial stability"
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
                "description": f"Large loan amount (${loan_amnt:,.0f}) - higher exposure"
            })
        
        # ============================================
        # STEP 8: Return prediction response
        # ============================================
        response = {
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'risk_factors': risk_factors
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f'Prediction error: {str(e)}'
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_status = 'loaded' if rf_model is not None else 'not_loaded'
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'timestamp': datetime.now().isoformat(),
        'service': 'Credit Risk Prediction API'
    })

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with API information"""
    model_status = 'loaded' if rf_model is not None else 'not_loaded'
    return jsonify({
        'service': 'Credit Risk Prediction API',
        'version': '2.0.0',
        'model': 'Random Forest Classifier',
        'model_status': model_status,
        'features_count': len(model_columns) if model_columns else 0,
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        },
        'status': 'running'
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run API server or retrain model')
    parser.add_argument('--train', action='store_true', help='Train model from CSV and save artifacts')
    parser.add_argument('--data', default='bankloans.csv', help='Path to training CSV file (used with --train)')
    parser.add_argument('--out', default='credit_risk_model', help='Output prefix for saved artifacts')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', default=5000, type=int)
    args = parser.parse_args()

    if args.train:
        print(f"Training requested. Using data: {args.data}")
        ok = train_model(args.data, args.out)
        if ok:
            print("Training finished successfully. Artifacts saved.")
        else:
            print("Training failed. See errors above.")
        # when training from CLI we exit after training
        exit(0 if ok else 1)

    # Not training -> start server (load artifacts first)
    load_artifacts()
    # run startup checks and diagnostic tests
    run_startup_diagnostic()

    print("\n" + "="*70)
    print("Credit Risk Prediction API with Random Forest")
    print("="*70)
    if rf_model is not None:
        print("✓ Model ready for predictions")
    else:
        print("✗ Model NOT loaded - please run training script first (python app.py --train --data path/to/file.csv)")
    print("="*70)
    print(f"Starting server on http://{args.host}:{args.port}")
    print("="*70 + "\n")
    app.run(debug=True, port=args.port, host=args.host)