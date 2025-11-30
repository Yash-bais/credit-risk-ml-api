import joblib
import pprint

print('Inspecting model artifacts in current directory...')

try:
    model = joblib.load('credit_risk_model.pkl')
    print('\nLoaded credit_risk_model.pkl')
    print('Type:', type(model))
    # print attrs safely
    attrs = {}
    if hasattr(model, 'n_features_in_'):
        attrs['n_features_in_'] = getattr(model, 'n_features_in_')
    if hasattr(model, 'feature_importances_'):
        attrs['feature_importances_len'] = len(getattr(model, 'feature_importances_', []))
    if hasattr(model, 'classes_'):
        attrs['classes_'] = list(getattr(model, 'classes_'))
    if hasattr(model, 'get_params'):
        attrs['params_keys'] = list(model.get_params().keys())[:10]
    pprint.pprint(attrs)
except Exception as e:
    print('Failed to load model:', e)

try:
    scaler = joblib.load('scaler.pkl')
    print('\nLoaded scaler.pkl')
    if hasattr(scaler, 'mean_'):
        print('scaler.mean_ shape:', getattr(scaler, 'mean_').shape)
    if hasattr(scaler, 'scale_'):
        print('scaler.scale_ shape:', getattr(scaler, 'scale_').shape)
except Exception as e:
    print('Failed to load scaler:', e)

try:
    cols = joblib.load('model_columns.pkl')
    print('\nLoaded model_columns.pkl -> length:', len(cols))
    print('First 30 columns (or all):')
    pprint.pprint(cols[:30])
except Exception as e:
    print('Failed to load model_columns:', e)

print('\nDone.')
