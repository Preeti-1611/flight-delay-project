import pandas as pd
import plotly.express as px
import plotly.io as pio
import os

def run_eda():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_paths = [
        os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_dataset.csv'),
        os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_enriched.csv'),
        os.path.join(base_dir, '..', 'data', 'raw', 'indian_flights_500_rows.csv')
    ]
    raw_path = next((path for path in raw_data_paths if os.path.exists(path)), raw_data_paths[0])
    charts_dir = os.path.join(base_dir, '..', 'dashboard', 'charts') # saving charts for dashboard
    os.makedirs(charts_dir, exist_ok=True)

    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    df = pd.read_csv(raw_path)
    
    # Map Indian flights columns to standard names
    column_mapping = {
        'distance_km': 'distance',
        'arrival_delay_minutes': 'arrival_delay'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Feature Engineering for EDA
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['is_delayed'] = (df['arrival_delay'] > 15).astype(int)

    # 1. Delay rate by airline
    airline_delay = df.groupby('airline')['is_delayed'].mean().reset_index()
    fig1 = px.bar(airline_delay, x='airline', y='is_delayed', title='Delay Rate by Airline', 
                  labels={'is_delayed': 'Delay Probability'})
    pio.write_json(fig1, os.path.join(charts_dir, 'delay_by_airline.json'))

    # 2. Delay rate by airport (Origin)
    origin_delay = df.groupby('origin')['is_delayed'].mean().reset_index()
    fig2 = px.bar(origin_delay, x='origin', y='is_delayed', title='Delay Rate by Origin Airport',
                  labels={'is_delayed': 'Delay Probability'})
    pio.write_json(fig2, os.path.join(charts_dir, 'delay_by_origin.json'))

    # 3. Delay by month
    month_delay = df.groupby('month')['is_delayed'].mean().reset_index()
    fig3 = px.line(month_delay, x='month', y='is_delayed', title='Delay Rate by Month',
                   labels={'is_delayed': 'Delay Probability'})
    pio.write_json(fig3, os.path.join(charts_dir, 'delay_by_month.json'))

    # 4. Delay by weekday
    weekday_delay = df.groupby('weekday')['is_delayed'].mean().reset_index()
    fig4 = px.bar(weekday_delay, x='weekday', y='is_delayed', title='Delay Rate by Weekday',
                  labels={'is_delayed': 'Delay Probability'})
    pio.write_json(fig4, os.path.join(charts_dir, 'delay_by_weekday.json'))
    
    # 5. Delay vs Departure Time
    # Binning departure time for better visualization
    df['hour'] = (df['departure_time'] // 100).astype(int)
    time_delay = df.groupby('hour')['is_delayed'].mean().reset_index()
    fig5 = px.line(time_delay, x='hour', y='is_delayed', title='Delay Rate by Time of Day')
    pio.write_json(fig5, os.path.join(charts_dir, 'delay_by_time.json'))

    # 6. Delay vs Distance
    # Validating relationship with scatter or binning
    # Scatter might be too dense, let's use scatter with trendline or binning
    # For speed on large datasets, maybe a hexbin or simple scatter on sample. 
    # Using simple scatter for now as dataset is small/medium.
    fig6 = px.scatter(df, x='distance', y='arrival_delay', title='Arrival Delay vs Distance', trendline="ols")
    pio.write_json(fig6, os.path.join(charts_dir, 'delay_vs_distance.json'))

    print(f"EDA charts saved to {charts_dir}")

if __name__ == "__main__":
    run_eda()
