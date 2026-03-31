import joblib
import pandas as pd
import numpy as np
import os

# Load models and encoders globally to avoid reloading on every call
base_dir = os.path.dirname(os.path.abspath(__file__))
classifier_path = os.path.join(base_dir, '..', 'trained_models', 'delay_classifier.pkl')
regressor_path = os.path.join(base_dir, '..', 'trained_models', 'delay_regressor.pkl')
encoders_path = os.path.join(base_dir, '..', 'trained_models', 'encoders.pkl')

classifier = None
regressor = None
encoders = None

def load_resources():
    global classifier, regressor, encoders
    if os.path.exists(classifier_path):
        classifier = joblib.load(classifier_path)
    if os.path.exists(regressor_path):
        regressor = joblib.load(regressor_path)
    if os.path.exists(encoders_path):
        encoders = joblib.load(encoders_path)

def predict_flight_delay(input_dict):
    """
    Predicts flight delay probability and estimated minutes.
    """
    if classifier is None or regressor is None or encoders is None:
        load_resources()
        if classifier is None or regressor is None or encoders is None:
            return {'error': "Models not trained yet. Please run the training pipeline first."}

    # Prepare input
    encoded_input = input_dict.copy()
    
    # Define features and ensure order matches training
    features = ['airline', 'origin', 'destination', 'month', 'day', 'weekday', 'departure_time', 'distance']
    
    # Categorical columns to encode
    cat_cols = ['airline', 'origin', 'destination']
    if 'weather_condition' in encoders:
        cat_cols.append('weather_condition')
        features.extend(['weather_condition', 'wind_speed', 'visibility'])

    try:
        for col in cat_cols:
            if col in encoders:
                le = encoders[col]
                val = input_dict.get(col)
                if val not in le.classes_:
                    raise ValueError(f"Unknown category '{val}' for field '{col}'")
                encoded_input[col] = le.transform([val])[0]
    except Exception as e:
        return {'error': str(e)}

    # Create feature vector
    input_df = pd.DataFrame([encoded_input], columns=features)
    
    # Predict
    prob = classifier.predict_proba(input_df)[0][1]
    delay_min = regressor.predict(input_df)[0]
    
    return {
        'probability': round(float(prob) * 100, 2),
        'predicted_delay': round(float(delay_min), 2)
    }
