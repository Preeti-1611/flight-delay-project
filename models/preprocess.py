import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data():
    # Define paths relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_paths = [
        os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_dataset.csv'),
        os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_enriched.csv'),
        os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_500_rows.csv')
    ]
    raw_path = next((path for path in raw_data_paths if os.path.exists(path)), raw_data_paths[0])
    
    processed_path = os.path.join(base_dir, '..', 'data', 'processed', 'cleaned_flights.csv')
    encoders_path = os.path.join(base_dir, '..', 'trained_models', 'encoders.pkl')

    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    print(f"Loading data from {raw_path}...")
    df = pd.read_csv(raw_path)
    
    # Map Indian flights columns to standard names
    column_mapping = {
        'distance_km': 'distance',
        'arrival_delay_minutes': 'arrival_delay'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Handle missing values
    df.dropna(inplace=True)

    # Extract date features
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday  # Monday=0, Sunday=6

    # Encode categorical columns
    categorical_cols = ['airline', 'origin', 'destination']
    if 'weather_condition' in df.columns:
        categorical_cols.append('weather_condition')
        
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Save encoders for later use in prediction
    os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
    joblib.dump(encoders, encoders_path)

    # Create target column
    df['delayed'] = (df['arrival_delay'] > 15).astype(int)

    # Save cleaned dataset
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Data preprocessing complete. Saved to {processed_path}")

if __name__ == "__main__":
    preprocess_data()
