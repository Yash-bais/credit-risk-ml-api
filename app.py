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

        # Map Floot form fields to model fields
        input_data = {
            'person_age': data.get('person_age'),
            'person_income': data.get('annualIncome'),
            'person_emp_length': data.get('employmentYears'),
            'loan_amnt': data.get('loanAmount'),
            'loan_int_rate': data.get('loan_int_rate', 10.0),  # default
            'loan_percent_income': data.get('loanAmount') / data.get('annualIncome') if data.get(
                'annualIncome') > 0 else 0,
            'cb_person_cred_hist_length': data.get('creditScore') / 100,  # normalize
            'person_home_ownership': data.get('person_home_ownership', 'RENT'),
            'loan_intent': data.get('loanPurpose', 'PERSONAL').upper(),
            'loan_grade': data.get('loan_grade', 'B'),
            'cb_person_default_on_file': 'Y' if data.get('existingDebt', 0) > data.get('annualIncome', 1) * 0.4 else 'N'
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Apply same preprocessing as training
        df = pd.get_dummies(df,
                            columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'],
                            drop_first=True)

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
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)