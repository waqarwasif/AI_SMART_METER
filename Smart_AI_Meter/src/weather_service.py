import requests
import pandas as pd
from datetime import datetime

def get_karachi_weather_forecast():
    """
    Fetches REAL-TIME Hourly Temperature for Karachi from Open-Meteo.
    FILTERS out past hours so the data starts exactly from the Current Hour.
    """
    print("☁️ Connecting to Weather Satellite (Open-Meteo)...")
    
    # Karachi Coordinates
    LAT = 24.8607
    LONG = 67.0011
    
    # API Call
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LONG}&hourly=temperature_2m&timezone=Asia%2FKarachi"
    
    try:
        response = requests.get(url, timeout=10) # 10s timeout
        response.raise_for_status() # Raise error if website is down
        data = response.json()
        
        # 1. Create the DataFrame from API Data
        # This raw list starts at 00:00 Midnight (The 16.4°C you saw)
        raw_df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['hourly']['time']),
            'temperature_c': data['hourly']['temperature_2m']
        })
        
        # 2. THE CRITICAL FIX: Filter for NOW
        # We check the current time and throw away any row older than "This Hour"
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # Keep only rows where timestamp is >= current hour
        real_time_df = raw_df[raw_df['timestamp'] >= current_hour].reset_index(drop=True)
        
        # 3. Take exactly 168 hours (7 Days)
        real_time_df = real_time_df.head(168)
        
        # 4. Verify we actually got data
        if len(real_time_df) == 0:
            raise ValueError("API Data was empty after filtering!")
            
        current_temp = real_time_df['temperature_c'].iloc[0]
        print(f"✅ Weather Data Received. Current Temp: {current_temp}°C")
        
        return real_time_df
        
    except Exception as e:
        print(f"❌ WEATHER API FAILED: {e}")
        # IMPORTANT: Returning None will trigger the backup. 
        # If you strictly WANT NO FAKE DATA, the app will stop here.
        return None