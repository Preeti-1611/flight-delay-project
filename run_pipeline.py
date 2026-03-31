import sys
import os


from models.preprocess import preprocess_data
from models.eda import run_eda
from models.train_classifier import train_classifier
from models.train_regressor import train_regressor

def run_pipeline():
    print("-----------------------------------------------------")
    print("STEP 1: Data Preprocessing")
    preprocess_data()
    
    print("\n-----------------------------------------------------")
    print("STEP 2: Exploratory Data Analysis")
    run_eda()
    
    print("\n-----------------------------------------------------")
    print("STEP 3: Training Classifier")
    train_classifier()
    
    print("\n-----------------------------------------------------")
    print("STEP 4: Training Regressor")
    train_regressor()
    
    print("\n-----------------------------------------------------")
    print("Pipeline Complete!")
    print("To run the dashboard: streamlit run dashboard/app.py")

if __name__ == "__main__":
    run_pipeline()
