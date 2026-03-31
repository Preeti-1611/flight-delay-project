import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Configuration
NUM_RECORDS = 5000
OUTPUT_FILE = r'c:\Users\PREETI NAGARALE\flight_delay_project\data\raw\indian_flights_enriched.csv'

# Data from existing dataset
airlines = [
    "Alliance Air", "Air India", "AirAsia India", "SpiceJet", 
    "Vistara Airlines", "Akasa Air", "IndiGo Airlines"
]
locations = [
    "Jaipur, Rajasthan", "Chennai, Tamil Nadu", "Pune, Maharashtra", 
    "Hyderabad, Telangana", "Bengaluru, Karnataka", "Kochi, Kerala", 
    "Ahmedabad, Gujarat", "Kolkata, West Bengal", "Delhi, Delhi", 
    "Mumbai, Maharashtra", "Goa, Goa", "Lucknow, Uttar Pradesh"
]

weather_conditions = ["Clear", "Rain", "Storm"]

def generate_data():
    start_date = datetime(2024, 1, 1)
    data = []

    for i in range(NUM_RECORDS):
        date = start_date + timedelta(days=random.randint(0, 365))
        airline = random.choice(airlines)
        origin = random.choice(locations)
        destination = random.choice(locations)
        while destination == origin:
            destination = random.choice(locations)
        
        departure_time = random.randint(500, 2300) 
        distance = random.randint(300, 2500)
        
        # Weather features
        condition = random.choices(weather_conditions, weights=[0.7, 0.2, 0.1])[0]
        wind_speed = round(random.uniform(0, 40), 1) # km/h
        visibility = round(random.uniform(1, 10), 1)
        
        # Delay injection logic
        # Base delay + Time-based congestion
        # Flights later in the day suffer from "cascading delays"
        time_hour = departure_time // 100
        congestion_delay = 0
        if time_hour >= 18: # Evening peak (6 PM - 11 PM)
            congestion_delay = random.uniform(10, 40)
        elif time_hour >= 12: # Afternoon
            congestion_delay = random.uniform(0, 15)
        elif time_hour <= 8: # Early morning - most punctual
            congestion_delay = random.uniform(-10, 5)
            
        base_delay = np.random.normal(5 + congestion_delay, 10)
        
        # Weather influence
        weather_delay = 0
        if condition == "Storm":
            weather_delay = random.uniform(30, 90)
            wind_speed = random.uniform(25, 50)
            visibility = random.uniform(0.5, 2)
        elif condition == "Rain":
            weather_delay = random.uniform(10, 30)
            wind_speed = random.uniform(10, 25)
            visibility = random.uniform(2, 5)
        else: # Clear
            weather_delay = random.uniform(-10, 10)
            wind_speed = random.uniform(0, 15)
            visibility = random.uniform(7, 10)
            
        # Wind speed influence
        if wind_speed > 30:
            weather_delay += (wind_speed - 30) * 2
            
        # Visibility influence
        if visibility < 2:
            weather_delay += (2 - visibility) * 20
            
        arrival_delay = int(base_delay + weather_delay)
        
        data.append([
            date.strftime('%d-%m-%Y'), airline, origin, destination, 
            departure_time, distance, arrival_delay,
            condition, wind_speed, visibility
        ])

    columns = [
        'date', 'airline', 'origin', 'destination', 
        'departure_time', 'distance_km', 'arrival_delay_minutes',
        'weather_condition', 'wind_speed', 'visibility'
    ]
    df = pd.DataFrame(data, columns=columns)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {NUM_RECORDS} records and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_data()
