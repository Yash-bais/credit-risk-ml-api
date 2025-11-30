import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def main(csv_path: str = 'bankloans.csv'):
    print('='*70)
    print('TRAINING CREDIT RISK MODEL')
    print('='*70)

    df = pd.read_csv(csv_path)
    print(f'✓ Loaded dataset: {df.shape}')
    print(f'Columns: {df.columns.tolist()}')

    # Detect target column
    possible_targets = ['loan_status', 'default', 'target', 'loan_default']
    target_col = next((t for t in possible_targets if t in df.columns), None)
    if target_col is None:
        raise SystemExit(f'No target column found. Expected one of {possible_targets}')

    print(f'Using target column: {target_col}')
    print(df[target_col].value_counts())

    # drop rows with missing target values
    df = df.dropna(subset=[target_col]).copy()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # handle missing input values: simple imputation so training doesn't break
    # - numeric columns: fill with median
    # - categorical columns: fill with 'MISSING'
    if X.isna().any().any():
        print('\n⚠️ Found missing values in input features; applying simple imputation')
        for col in X.columns:
            if X[col].dtype.kind in 'biufc':
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna('MISSING')

    # detect categorical
    candidate_categorical = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    categorical_cols = [c for c in candidate_categorical if c in X.columns]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    print(f'\nNumerical features ({len(numerical_cols)}): {numerical_cols}')
    print(f'Categorical features ({len(categorical_cols)}): {categorical_cols}')

    numeric_transformer = StandardScaler()
    cat_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') if categorical_cols else None

    transformers = [('num', numeric_transformer, numerical_cols)]
    if categorical_cols:
        transformers.append(('cat', cat_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    print('\nFitting preprocessor...')
    X_transformed = preprocessor.fit_transform(X)

    num_feature_names = numerical_cols
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols) if categorical_cols else []
    all_feature_names = list(num_feature_names) + list(cat_feature_names)

    print(f'\n✓ Total features after encoding: {len(all_feature_names)}')

    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)
    print(f'\n✓ Train set: {X_train.shape}  Test set: {X_test.shape}')

    print('\nTraining RandomForest...')
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    print('\nEvaluating...')
    print('Training accuracy:', rf.score(X_train, y_train))
    print('Test accuracy:', rf.score(X_test, y_test))

    pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', rf)])
    joblib.dump(pipeline, 'credit_risk_model_pipeline.pkl')
    joblib.dump(rf, 'credit_risk_model.pkl')
    joblib.dump(all_feature_names, 'credit_risk_model_columns.pkl')
    joblib.dump(preprocessor.named_transformers_['num'], 'credit_risk_model_scaler.pkl')

    print('\nSaved artifacts:')
    print(' - credit_risk_model_pipeline.pkl')
    print(' - credit_risk_model.pkl')
    print(' - credit_risk_model_columns.pkl')
    print(' - credit_risk_model_scaler.pkl')

    # Test cases aligned to dataset schema where possible
    print('\n' + '='*70)
    print('TESTING MODEL WITH BAD AND GOOD APPLICATIONS')
    print('='*70)

    # Build two test rows using the dataset's columns if possible
    def make_row_from_cols(cols, defaults):
        row = {}
        for c in cols:
            row[c] = defaults.get(c, 0)
        return pd.DataFrame([row])

    # Defaults for a BAD applicant (low income, high loan)
    bad_defaults = {
        'age': 25, 'ed': 1, 'employ': 1, 'address': 1, 'income': 5000,
        'debtinc': 3000.0, 'creddebt': 1.0, 'othdebt': 150000
    }

    # Defaults for a GOOD applicant (high income, small loan)
    good_defaults = {
        'age': 35, 'ed': 3, 'employ': 10, 'address': 1, 'income': 100000,
        'debtinc': 10.0, 'creddebt': 15.0, 'othdebt': 10000
    }

    # Use all_feature_names to try to create a test vector that matches training
    # If some of the original raw columns don't match the canonical names, use numeric_cols
    raw_cols = numerical_cols.copy()
    bad_df = make_row_from_cols(raw_cols, bad_defaults)
    good_df = make_row_from_cols(raw_cols, good_defaults)

    for label, test_df in [('BAD', bad_df), ('GOOD', good_df)]:
        X_t = preprocessor.transform(test_df)
        pred = rf.predict(X_t)[0]
        proba = rf.predict_proba(X_t)[0]
        print(f"\n{label} -> Raw pred: {pred}, Proba: {proba}, Features sum: {X_t.sum():.2f}")

    print('\n✓ TRAINING COMPLETE - Ready to deploy!')

if __name__ == '__main__':
    main()