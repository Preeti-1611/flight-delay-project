# Flight Delay Prediction & Airline Analytics Dashboard

This project implements an end-to-end machine learning pipeline to predict flight delays and provide analytics on airline performance. It includes data preprocessing, exploratory data analysis, classification and regression modeling, and a Streamlit dashboard.

## Project Structure

```
flight-delay-predictor/
├── data/               # Raw and processed data
├── models/             # Python scripts for ML pipeline
│   ├── preprocess.py       # Cleans data & encodes features
│   ├── eda.py              # Generates analysis charts
│   ├── train_classifier.py # Trains classification models (Delayed Y/N)
│   ├── train_regressor.py  # Trains regression models (Delay Minutes)
│   └── predict.py          # Prediction logic
├── dashboard/          # Streamlit application
│   └── app.py
├── trained_models/     # Saved models (.pkl)
├── run_pipeline.py     # Script to run the full pipeline
├── requirements.txt    # Project dependencies
├── Procfile            # Deployment configuration
└── README.md           # Project documentation
```

## How to Run Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data** (if needed)
   ```bash
   python generate_data.py
   ```

3. **Run the ML Pipeline**
   This will preprocess data, run EDA, and train models.
   ```bash
   python run_pipeline.py
   ```

4. **Launch the Dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

## How to Deploy on Render

1. Create a new Web Service on Render.
2. Connect your GitHub repository.
3. Use the following settings:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard/app.py`
4. Deploy!

## Models Used

- **Classification**: Logistic Regression, Random Forest, XGBoost (Optimized for F1-score)
- **Regression**: Linear Regression, Random Forest Regressor (Optimized for RMSE/R2)
- **EDA**: Plotly charts for Airline, Airport, and Temporal analysis.

## Dataset
Synthetic dataset with columns: `date`, `airline`, `origin`, `destination`, `departure_time`, `distance`, `arrival_delay`.
Target logic: Delayed if `arrival_delay` > 15 minutes.
