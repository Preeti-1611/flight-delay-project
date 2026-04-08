import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_classifier():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'processed', 'cleaned_flights.csv')
    model_path = os.path.join(base_dir, '..', 'trained_models', 'delay_classifier.pkl')
    metrics_path = os.path.join(base_dir, '..', 'trained_models', 'classifier_metrics.json')

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(data_path)
    
    # Define features and target
    features = ['airline', 'origin', 'destination', 'month', 'day', 'weekday', 'departure_time', 'distance']
    if 'weather_condition' in df.columns:
        features.extend(['weather_condition', 'wind_speed', 'visibility'])
        
    target = 'delayed'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    best_model = None
    best_score = 0
    results = {}

    print("Training Classifiers...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        results[name] = {
            'Accuracy': round(acc, 4), 
            'Precision': round(prec, 4), 
            'Recall': round(rec, 4), 
            'F1': round(f1, 4),
            'Confusion_Matrix': cm
        }
        print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
        
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_model_name = name

    results['_best_model'] = best_model_name
    print(f"\nBest Model by F1 Score: {best_model.__class__.__name__}")
    
    # Save best model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save comparison metrics
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train_classifier()
