import joblib
import pandas as pd

print('Loading pipeline and sample data...')
pipe = joblib.load('credit_risk_model_pipeline.pkl')
df = pd.read_csv('bankloans.csv')
# drop rows with missing target and keep only features used by the pipeline
if 'default' in df.columns:
    target_col = 'default'
elif 'loan_status' in df.columns:
    target_col = 'loan_status'
else:
    target_col = None

if target_col:
    X = df.drop(columns=[target_col])
else:
    X = df

# Use first row
sample = X.iloc[[0]]
print('Sample features:', sample.iloc[0].to_dict())
if hasattr(pipe, 'predict_proba'):
    print('Predict_proba:', pipe.predict_proba(sample))
print('Predict:', pipe.predict(sample))
