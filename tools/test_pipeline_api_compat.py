"""Quick test: load pipeline and ensure person_* and raw numeric payloads produce different predictions.
Run: python tools/test_pipeline_api_compat.py
"""
import joblib
import pandas as pd

PIPELINE_FN = 'credit_risk_model_pipeline.pkl'

print('Loading pipeline...')
pipeline = joblib.load(PIPELINE_FN)
print('Loaded pipeline')

# helper from app.py mapping
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

# get expected raw cols
preproc = pipeline.named_steps['preprocessor']
expected_raw_cols = []
for _name, _trans, cols in preproc.transformers:
    if isinstance(cols, (list, tuple)):
        expected_raw_cols.extend(list(cols))
expected_raw_cols = list(dict.fromkeys(expected_raw_cols))
print('Expected raw cols:', expected_raw_cols)

# two payloads
person_payload = {
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

raw_payload = {
    'age': 35,
    'ed': 3,
    'employ': 10,
    'address': 1,
    'income': 100000,
    'debtinc': 10.0,
    'creddebt': 15.0,
    'othdebt': 10000
}

# mapping helper
def map_to_raw_columns(raw_cols, payload_row: dict):
    row = {}
    for col in raw_cols:
        val = None
        if col in payload_row:
            val = payload_row.get(col)
        elif col in mapping_alternatives:
            for alt in mapping_alternatives[col]:
                if alt in payload_row:
                    val = payload_row.get(alt)
                    break
        if val is None:
            val = 0
        row[col] = val
    return pd.DataFrame([row])

p1 = map_to_raw_columns(expected_raw_cols, person_payload)
p2 = map_to_raw_columns(expected_raw_cols, raw_payload)

print('\nMapped person payload:', p1.to_dict(orient='records'))
print('\nMapped raw payload:', p2.to_dict(orient='records'))

proba1 = pipeline.predict_proba(p1)[0]
proba2 = pipeline.predict_proba(p2)[0]

print('\nPerson-style prediction proba:', proba1)
print('Raw-style prediction proba:', proba2)

assert not (proba1 == proba2).all(), 'PROBABILITIES IDENTICAL — encoding mismatch or mapping failure'
print('\n✅ Test passed — probabilities differ as expected')
