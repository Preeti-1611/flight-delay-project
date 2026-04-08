import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_regressor():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'processed', 'cleaned_flights.csv')
    model_path = os.path.join(base_dir, '..', 'trained_models', 'delay_regressor.pkl')
    metrics_path = os.path.join(base_dir, '..', 'trained_models', 'regressor_metrics.json')

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
        
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        
        results[name] = {'RMSE': round(rmse, 4), 'R2': round(r2, 4), 'MAE': round(mae, 4)}
        print(f"{name}: RMSE={rmse:.4f}, R2={r2:.4f}, MAE={mae:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_model_name = name

    results['_best_model'] = best_model_name
    print(f"\nBest Model by R2 Score: {best_model.__class__.__name__}")
    
    # Save best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save comparison metrics
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_regressor()
