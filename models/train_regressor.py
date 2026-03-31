import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_regressor():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'processed', 'cleaned_flights.csv')
    model_path = os.path.join(base_dir, '..', 'trained_models', 'delay_regressor.pkl')

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Define features and target (Predict actual delay in minutes)
    features = ['airline', 'origin', 'destination', 'month', 'day', 'weekday', 'departure_time', 'distance']
    if 'weather_condition' in df.columns:
        features.extend(['weather_condition', 'wind_speed', 'visibility'])
        
    target = 'arrival_delay'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = -float('inf') # R2 can be negative
    results = {}

    print("Training Regressors...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'RMSE': rmse, 'R2': r2}
        print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model

    print(f"\nBest Model by R2 Score: {best_model.__class__.__name__}")
    
    # Save best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_regressor()
