import pandas as pd
import numpy as np
import os
from src.weather_service import get_karachi_weather_forecast

def generate_future_features(model_features):
    # 1. Calling the API
    weather_df = get_karachi_weather_forecast()
    
    # 2.CHECK: Did we get real data?
    if weather_df is None:
        print("\n" + "!"*50)
        print("   Check your Internet Connection!")
        print("!"*50 + "\n")
        
        # Backup Simulation (Only runs if API fails)
        from datetime import datetime, timedelta
        start_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        future_hours = [start_date + timedelta(hours=i) for i in range(168)]
        weather_df = pd.DataFrame({'timestamp': future_hours})
        weather_df['temperature_c'] = 25.0 
    
    # 3. Add Math Features (Sin/Cos)
    df = weather_df.copy()
    
    # Extract standard time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    
    # Add Cyclical Features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Add any other features the model expects
    if 'is_weekend' in model_features:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
    if 'week_of_month' in model_features:
        df['week_of_month'] = df['day_of_month'].apply(lambda d: (d - 1) // 7 + 1)

    if 'day_sin' in model_features:
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    if 'month_sin' in model_features:
        df['month'] = df['timestamp'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

def predict_next_week(model, feature_cols):
    print("\n🔮 Generating Forecast...")
    
    # 1. Build Features
    future_df = generate_future_features(feature_cols)
    
    # 2. Align columns for the Model
    X_future = future_df[feature_cols]
    
    # 3. Predict
    future_df['predicted_usage_kwh'] = model.predict(X_future)
    
    
    return future_df