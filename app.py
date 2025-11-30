from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from your Floot app

# Load saved model artifacts
model = joblib.load('credit_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Safely get values with defaults to prevent None comparison errors
        annual_income = data.get('person_income', 0) or 0
        loan_amount = data.get('loan_amnt', 0) or 0
        
        # Map Floot form fields to model fields
        input_data = {
            'person_age': data.get('person_age', 30),
            'person_income': annual_income,
            'person_emp_length': data.get('person_emp_length', 0),
            'loan_amnt': loan_amount,
            'loan_int_rate': data.get('loan_int_rate', 10.0),
            'loan_percent_income': loan_amount / annual_income if annual_income > 0 else 0,
            'cb_person_cred_hist_length': data.get('cb_person_cred_hist_length', 5.0),
            'person_home_ownership': data.get('person_home_ownership', 'RENT'),
            'loan_intent': data.get('loan_intent', 'PERSONAL').upper(),
            'loan_grade': data.get('loan_grade', 'B'),
            'cb_person_default_on_file': data.get('cb_person_default_on_file', 'N')
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Apply same preprocessing as training
        df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], drop_first=True)
        
        # Align columns with training data
        df = df.reindex(columns=model_columns, fill_value=0)
        
        # Scale
        scaled_data = scaler.transform(df)
        
        # Predict
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]
        confidence = float(max(probabilities))
        
        # Get feature importances for risk factors
        feature_importance = model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-5:][::-1]
        risk_factors = [model_columns[i] for i in top_features_idx]
        
        return jsonify({
            'predictionResult': 'rejected' if prediction == 1 else 'approved',
            'approvalConfidence': confidence,
            'riskFactors': risk_factors,
            'message': 'Prediction completed successfully'
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # This will show in Render logs
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)