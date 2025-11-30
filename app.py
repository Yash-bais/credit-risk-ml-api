from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================
# LOAD MODEL ARTIFACTS AT STARTUP
# ============================================
print("Loading model artifacts...")
try:
    rf_model = joblib.load('credit_risk_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    print(f"✓ Model loaded successfully")
    print(f"✓ Model expects {len(model_columns)} features: {model_columns}")
except FileNotFoundError as e:
    print(f"ERROR: Model files not found. Please run the training script first.")
    print(f"Missing file: {e}")
    rf_model = None
    scaler = None
    model_columns = None

if rf_model is not None:
    print("\n" + "="*60)
    print("TESTING MODEL WITH BAD APPLICATION")
    print("="*60)
    
    # Test with obviously bad application
    test_bad = {
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
    
    # Encode it the same way
    df_test = pd.DataFrame([test_bad])
    df_test_encoded = pd.get_dummies(
        df_test, 
        columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'],
        drop_first=True
    )
    
    # Add missing columns
    for col in model_columns:
        if col not in df_test_encoded.columns:
            df_test_encoded[col] = 0
    df_test_aligned = df_test_encoded[model_columns]
    
    # Scale and predict
    X_test_scaled = scaler.transform(df_test_aligned)
    prediction_result = rf_model.predict(X_test_scaled)[0]
    prediction_proba = rf_model.predict_proba(X_test_scaled)[0]
    
    print(f"Bad application raw prediction: {prediction_result}")
    print(f"Prediction probabilities: {prediction_proba}")
    print(f"Expected: 1 (means default/reject)")
    print(f"If you got 0, your labels are FLIPPED!")
    print("="*60 + "\n")

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
        print(f"Received prediction request: {data}")
        
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
        
        print(f"Input dataframe before encoding:\n{df_input}")
        
        # ============================================
        # STEP 3: Apply one-hot encoding (same as training)
        # ============================================
        df_encoded = pd.get_dummies(
            df_input, 
            columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'],
            drop_first=True
        )
        
        print(f"Encoded features: {list(df_encoded.columns)}")
        
        # ============================================
        # STEP 4: Align features with training data
        # ============================================
        # Add missing columns with 0 values
        for col in model_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # Keep only the columns that were in training (in the same order)
        df_aligned = df_encoded[model_columns]
        
        print(f"Aligned features shape: {df_aligned.shape}")
        print(f"Feature values:\n{df_aligned.iloc[0].to_dict()}")
        
        # ============================================
        # STEP 5: Scale features
        # ============================================
        X_scaled = scaler.transform(df_aligned)
        
        # ============================================
        # STEP 6: Make prediction
        # ============================================
        prediction_numeric = rf_model.predict(X_scaled)[0]
        prediction_proba = rf_model.predict_proba(X_scaled)[0]
        
        # prediction_numeric: 0 = approved (no default), 1 = rejected (default likely)
        prediction = "approved" if prediction_numeric == 1 else "rejected"
        confidence = float(prediction_proba.max())  # Confidence in the prediction
        
        print(f"Raw prediction: {prediction_numeric}")
        print(f"Prediction probabilities: {prediction_proba}")
        print(f"Final prediction: {prediction}, Confidence: {confidence:.4f}")
        
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
        
        print(f"Returning response: {response}")
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
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)'
        },
        'status': 'running'
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Credit Risk Prediction API with Random Forest")
    print("=" * 60)
    if rf_model is not None:
        print("✓ Model loaded successfully")
        print(f"✓ Ready to make predictions")
    else:
        print("✗ Model NOT loaded - please run training script first")
    print("=" * 60)
    print("Starting server on http://0.0.0.0:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')