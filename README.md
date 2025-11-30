# Credit Risk Prediction API

This project is a small Flask API that serves a Random Forest credit risk model.

Key files:
- `app.py` — Flask API, includes training CLI (`--train`) and prediction endpoint `/predict`.
- `bankloans.csv` — example training data (already provided in repo).
- `requirements.txt` — dependencies.
- `Procfile` — runs `gunicorn app:app` for deployment platforms like Heroku.

## Quick local workflow

1) Install dependencies and create a virtual environment (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

2) Train the model (place your CSV next to `app.py` or pass full path):

```powershell
# default expects bankloans.csv in project root
python app.py --train --data bankloans.csv
```

This will save the artifacts to the project folder:
- `credit_risk_model.pkl` (model)
- `credit_risk_model_scaler.pkl` (scaler)
- `credit_risk_model_columns.pkl` (feature list)
- `credit_risk_model_pipeline.pkl` (recommended pipeline — scaler + model)

3) Run the server:

```powershell
python app.py
# or change host/port
python app.py --host 127.0.0.1 --port 5000
```

4) Test endpoints
- Health: GET `/health`
- Predict: POST `/predict` with JSON body.

Example request formats accepted by `/predict`:

- Numeric schema (preferred for the trained `bankloans.csv` model):
```
{
  "age": 41,
  "ed": 3,
  "employ": 17,
  "address": 12,
  "income": 176,
  "debtinc": 9.3,
  "creddebt": 11.359392,
  "othdebt": 5.008608
}
```

- Floot/app style mapping (API accepts these and will try to translate them to the numeric features):
```
{
  "person_age": 41,
  "person_income": 176,
  "person_emp_length": 17,
  "loan_amnt": 5000,
  "loan_percent_income": 5.0,
  "creditScore": 700
}
```

Notes
- The `--train` CLI will retrain the model. For production it's recommended
  to train offline, save the pipeline, and deploy the pipeline artifact.
- The API prefers `credit_risk_model_pipeline.pkl` (preprocessing + model); this
  prevents feature alignment errors.

## Deploying / integrating with Floot

- You can deploy this app to Heroku or any host that supports WSGI/Gunicorn.
- The included `Procfile` will work on Heroku: `web: gunicorn app:app`.
- Once deployed, copy the `/predict` URL and configure your Floot app to POST
  either the numeric schema or the Floot form fields — the API attempts to map
  common Floot field names to the model's expected numeric features.

If you'd like, I can: add a `ColumnTransformer` into training for a fully baked
preprocessing pipeline, update the API to validate inputs more strictly, or
prepare a production-ready Dockerfile / CI pipeline for you.
