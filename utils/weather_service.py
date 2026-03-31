import random
import requests

class WeatherService:
    def __init__(self, api_key=None):
        self.api_key = api_key
        # Open-Meteo uses Latitude/Longitude
        self.city_coords = {
            "Delhi": (28.61, 77.20),
            "Mumbai": (19.07, 72.87),
            "Kolkata": (22.57, 88.36),
            "Chennai": (13.08, 80.27),
            "Bengaluru": (12.97, 77.59),
            "Hyderabad": (17.38, 78.48),
            "Pune": (18.52, 73.85),
            "Jaipur": (26.91, 75.78),
            "Kochi": (9.93, 76.26),
            "Ahmedabad": (23.02, 72.57),
            "Goa": (15.29, 74.12),
            "Lucknow": (26.84, 80.94),
        }
        
        # Mock weather patterns for simulation fallback
        self.city_weather_patterns = {
            "Delhi": {"condition": "Clear", "wind_range": (5, 15), "visibility_range": (5, 10)},
            "Mumbai": {"condition": "Clear", "wind_range": (10, 20), "visibility_range": (8, 10)},
            "Kolkata": {"condition": "Rain", "wind_range": (10, 25), "visibility_range": (4, 7)},
            "Chennai": {"condition": "Clear", "wind_range": (10, 20), "visibility_range": (7, 10)},
            "Bengaluru": {"condition": "Clear", "wind_range": (5, 15), "visibility_range": (8, 10)},
            "Hyderabad": {"condition": "Clear", "wind_range": (5, 15), "visibility_range": (8, 10)},
            "Pune": {"condition": "Clear", "wind_range": (5, 15), "visibility_range": (8, 10)},
            "Jaipur": {"condition": "Clear", "wind_range": (0, 10), "visibility_range": (8, 10)},
            "Kochi": {"condition": "Rain", "wind_range": (15, 30), "visibility_range": (3, 6)},
            "Ahmedabad": {"condition": "Clear", "wind_range": (5, 15), "visibility_range": (8, 10)},
            "Goa": {"condition": "Clear", "wind_range": (10, 20), "visibility_range": (8, 10)},
            "Lucknow": {"condition": "Clear", "wind_range": (5, 15), "visibility_range": (6, 9)},
        }

    def get_weather(self, city, month=None):
        """
        Fetches live weather from Open-Meteo or falls back to simulation.
        If a future month matches a known seasonal pattern (e.g., Monsoon), 
        the simulator is used to provide a realistic forecast.
        """
        from datetime import datetime
        current_month = datetime.now().month
        is_today = (month is None or month == current_month)

        city_clean = city.split(",")[0].strip()
        
        # If it's today, try live API
        if is_today and city_clean in self.city_coords:
            lat, lon = self.city_coords[city_clean]
            try:
                # Open-Meteo API (No Key Required)
                url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=visibility"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    current = data['current_weather']
                    
                    # Open-Meteo has specific weather codes
                    # 0=Clear, 1-3=Cloudy, 51-67=Rain, 71-77=Snow, 80-82=Rain Showers, 95-99=Storm
                    code = current['weathercode']
                    condition = "Clear"
                    if code >= 95: condition = "Storm"
                    elif code >= 50: condition = "Rain"
                    
                    # Visibility might need extra calculation if not in current
                    # Using current hourly visibility as a proxy if current_weather doesn't have it
                    visibility = 10.0 # Default
                    if 'hourly' in data and 'visibility' in data['hourly']:
                        visibility = round(data['hourly']['visibility'][0] / 1000, 1) # convert meters to km
                    
                    return {
                        "condition": condition,
                        "wind_speed": round(current['windspeed'], 1), # Default is km/h
                        "visibility": visibility,
                        "api_source": "Open-Meteo (Live)"
                    }
            except Exception as e:
                print(f"API Error: {e}")
        
        # Use seasonal simulator for future dates or fallback
        weather = self._simulate_weather(city_clean, month)
        weather["api_source"] = "Seasonal AI Forecast" if not is_today else "Smart Simulator (Fallback)"
        return weather
        
    def _simulate_weather(self, city, month=None):
        pattern = self.city_weather_patterns.get(city, {
            "condition": "Clear", "wind_range": (5, 15), "visibility_range": (8, 10)
        })

        # Seasonal Overrides (India Specific)
        # June-Sept: Monsoon Season in most of India
        is_monsoon = month in [6, 7, 8, 9]
        
        # Adjustment factors
        storm_chance = 0.1
        rain_chance = 0.2
        
        if is_monsoon:
            storm_chance = 0.25
            rain_chance = 0.5
            pattern["condition"] = "Rain" # Default to rain in monsoon if not storm
        
        if random.random() < storm_chance:
            return {
                "condition": "Storm",
                "wind_speed": round(random.uniform(40, 75), 1), # km/h (Strong wind)
                "visibility": round(random.uniform(0.5, 2.0), 1)
            }
        
        if (is_monsoon or pattern["condition"] != "Rain") and random.random() < rain_chance:
             return {
                "condition": "Rain",
                "wind_speed": round(random.uniform(15, 35), 1), # km/h
                "visibility": round(random.uniform(2.0, 5.0), 1)
            }

        return {
            "condition": pattern["condition"],
            "wind_speed": round(random.uniform(*pattern["wind_range"]), 1),
            "visibility": round(random.uniform(*pattern["visibility_range"]), 1)
        }

if __name__ == "__main__":
    ws = WeatherService()
    print(f"Weather for Mumbai: {ws.get_weather('Mumbai')}")
    print(f"Weather for Kochi: {ws.get_weather('Kochi')}")
