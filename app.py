from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
# your ML imports here

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract ALL fields from the request
        features = {
            # Original fields
            'annual_income': float(data.get('annualIncome', 0)),
            'loan_amount': float(data.get('loanAmount', 0)),
            'credit_score': int(data.get('creditScore', 300)),
            'existing_debt': float(data.get('existingDebt', 0)),
            'employment_years': int(data.get('employmentYears', 0)),
            'employment_status': data.get('employmentStatus', 'unemployed'),
            'loan_purpose': data.get('loanPurpose', 'other'),
            'has_collateral': data.get('hasCollateral', False),
            
            # NEW FIELDS - Add these!
            'person_age': int(data.get('personAge', 18)),
            'home_ownership': data.get('homeOwnership', 'rent'),
            'loan_term': int(data.get('loanTerm', 36)),
            'credit_history_length': int(data.get('creditHistoryLength', 0)),
            'has_previous_defaults': data.get('hasPreviousDefaults', False),
            'loan_grade': data.get('loanGrade', 'C'),
            'interest_rate': float(data.get('interestRate', 10.0))
        }
        
        # Convert to DataFrame for your ML model
        df = pd.DataFrame([features])
        
        # YOUR ML PREDICTION CODE HERE
        # prediction = your_model.predict(df)
        # confidence = your_model.predict_proba(df)
        
        # For now, example response structure:
        prediction_result = "approved"  # or "rejected"
        confidence = 0.93  # 0 to 1
        
        # Generate risk factors based on the data
        risk_factors = []
        
        # Example risk factor logic
        if features['credit_score'] < 650:
            risk_factors.append({
                "factor": "credit_score",
                "impact": "negative",
                "description": "Credit score below optimal threshold"
            })
        elif features['credit_score'] >= 750:
            risk_factors.append({
                "factor": "credit_score",
                "impact": "positive",
                "description": "Excellent credit score"
            })
            
        dti_ratio = (features['existing_debt'] / features['annual_income']) * 100
        if dti_ratio > 40:
            risk_factors.append({
                "factor": "dti_ratio",
                "impact": "negative",
                "description": f"High debt-to-income ratio ({dti_ratio:.1f}%)"
            })
        elif dti_ratio < 30:
            risk_factors.append({
                "factor": "dti_ratio",
                "impact": "positive",
                "description": f"Low debt-to-income ratio ({dti_ratio:.1f}%)"
            })
            
        if features['has_previous_defaults']:
            risk_factors.append({
                "factor": "defaults",
                "impact": "negative",
                "description": "Previous payment defaults on record"
            })
            
        if features['employment_years'] > 3:
            risk_factors.append({
                "factor": "employment",
                "impact": "positive",
                "description": "Stable employment history"
            })
        
        return jsonify({
            'prediction': prediction_result,
            'confidence': confidence,
            'risk_factors': risk_factors
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)