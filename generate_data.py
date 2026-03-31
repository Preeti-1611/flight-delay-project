import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Create synthetic data
airlines = ['AA', 'UA', 'DL', 'WN', 'B6']
origins = ['JFK', 'LGA', 'EWR', 'ORD', 'LAX']
destinations = ['SFO', 'MIA', 'ATL', 'DFW', 'DEN']

start_date = datetime(2023, 1, 1)
data = []

for i in range(100):
    date = start_date + timedelta(days=random.randint(0, 365))
    airline = random.choice(airlines)
    origin = random.choice(origins)
    destination = random.choice(destinations)
    while destination == origin:
        destination = random.choice(destinations)
    
    departure_time = random.randint(600, 2200) # HHMM format approx
    distance = random.randint(200, 3000)
    arrival_delay = int(np.random.normal(10, 30)) # Mean delay 10 mins, std dev 30
    
    data.append([date.strftime('%Y-%m-%d'), airline, origin, destination, departure_time, distance, arrival_delay])

df = pd.DataFrame(data, columns=['date', 'airline', 'origin', 'destination', 'departure_time', 'distance', 'arrival_delay'])
df.to_csv(r'c:\Users\PREETI NAGARALE\.gemini\antigravity\playground\luminescent-crab\flight-delay-predictor\data\raw\flights.csv', index=False)
print("Synthetic data created.")
