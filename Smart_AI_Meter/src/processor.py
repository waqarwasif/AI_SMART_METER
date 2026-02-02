import numpy as np
import pandas as pd

def clean_data(input_df):
    """
    The 'Data Factory': Prepares raw user uploads for AI processing.
    Implements Physics-based constraints, Statistical Cleaning, and Feature Engineering.
    """
    
    df = input_df.copy()

    # 1. Standardization: Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    
    rename_map = {
        'usage': 'usage_kwh', 'kwh': 'usage_kwh',
        'temp': 'temperature_c', 'temperature': 'temperature_c', 'temp_c': 'temperature_c',
        'datetime': 'timestamp', 'time': 'timestamp'
    }
    df = df.rename(columns=rename_map)

    # --- CRITICAL FIX: TIME EXTRACTION ---
    # We must convert timestamp to datetime objects to extract hour/day/month
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Drop rows where timestamp failed to parse
        df = df.dropna(subset=['timestamp'])
        
        # Extract components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        
        # Sort by time to ensure linear interpolation works correctly
        df = df.sort_values('timestamp').reset_index(drop=True)

    # 2. THE CIRCUIT BREAKER (Physics Logic)
    HARD_LIMIT = 20.0 
    if 'usage_kwh' in df.columns:
        df['usage_kwh'] = df['usage_kwh'].clip(upper=HARD_LIMIT)

    # 3. Domain Integrity Checks
    if 'hour' in df.columns: df['hour'] = df['hour'].clip(0, 23)
    if 'temperature_c' in df.columns: df['temperature_c'] = df['temperature_c'].clip(5, 50) 
    if 'usage_kwh' in df.columns: df['usage_kwh'] = df['usage_kwh'].clip(lower=0)

    # 4. Logical Consistency
    if 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    if 'day_of_month' in df.columns:
        df['week_of_month'] = df['day_of_month'].apply(lambda d: (d - 1) // 7 + 1)

    # 5. Missing Value Imputation
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.fillna(0)

    # 6. Statistical Cleaning (Winsorization)
    if 'usage_kwh' in df.columns:
        upper_limit = df['usage_kwh'].quantile(0.99)
        if len(df) > 0:
            df['usage_kwh'] = df['usage_kwh'].clip(upper=upper_limit)

    # 7. Advanced Feature Engineering (Cyclical Encoding)
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)    

    if 'day_of_month' in df.columns:
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)

    print("✅ Data Cleaning Pipeline Complete.")
    return df