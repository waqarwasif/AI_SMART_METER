import numpy as np
import pandas as pd

def clean_data(input_df):
    """
    The 'Data Factory': Prepares raw user uploads for AI processing.
    Implements Physics-based constraints, Statistical Cleaning, and Feature Engineering.
    """
    # Best Practice: Work on a copy to prevent SettingWithCopy warnings
    df = input_df.copy()

    # 1. Standardization: Normalize column names to prevent KeyError
    df.columns = [c.strip().lower() for c in df.columns]
    
    rename_map = {
        'usage': 'usage_kwh', 'kwh': 'usage_kwh',
        'temp': 'temperature_c', 'temperature': 'temperature_c', 'temp_c': 'temperature_c'
    }
    df = df.rename(columns=rename_map)

    # 2. THE CIRCUIT BREAKER (Physics Logic)
    # Critical Safety Net: Catch physically impossible values (sensor glitches) before processing.
    # 20 kWh/hr is the hard limit for a residential main breaker.
    HARD_LIMIT = 20.0 
    
    if 'usage_kwh' in df.columns:
        impossible_count = (df['usage_kwh'] > HARD_LIMIT).sum()
        if impossible_count > 0:
            print(f"🔥 Found {impossible_count} impossible values (> {HARD_LIMIT} kWh). Clipping to reality.")
            df['usage_kwh'] = df['usage_kwh'].clip(upper=HARD_LIMIT)

    # 3. Domain Integrity Checks (Sanity Checks)
    # Enforce realistic bounds for time and physics
    if 'hour' in df.columns: df['hour'] = df['hour'].clip(0, 23)
    if 'day_of_week' in df.columns: df['day_of_week'] = df['day_of_week'].clip(0, 6)
    if 'temperature_c' in df.columns: df['temperature_c'] = df['temperature_c'].clip(5, 50) 
    if 'usage_kwh' in df.columns: df['usage_kwh'] = df['usage_kwh'].clip(lower=0)

    # 4. Logical Consistency (Data Integrity)
    # Auto-correct derivative features to ensure they match the primary data
    if 'day_of_week' in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    if 'day_of_month' in df.columns:
        df['week_of_month'] = df['day_of_month'].apply(lambda d: (d - 1) // 7 + 1)

    # 5. Missing Value Imputation
    # Use Linear Interpolation to preserve Time-Series continuity (better than filling with 0)
    df = df.interpolate(method='linear', limit_direction='both')
    df = df.fillna(0)

    # 6. Statistical Cleaning (Winsorization)
    # Cap extreme legitimate outliers at the 99th percentile to stabilize model training
    if 'usage_kwh' in df.columns:
        upper_limit = df['usage_kwh'].quantile(0.99)
        if len(df) > 0:
            outliers = (df['usage_kwh'] > upper_limit).sum()
            if outliers > 0:
                print(f"⚠️  Winsorized {outliers} statistical outliers > {upper_limit:.2f} kWh")
            df['usage_kwh'] = df['usage_kwh'].clip(upper=upper_limit)

    # 7. Advanced Feature Engineering (Cyclical Encoding)
    # Transform linear time (0-23) into circular coordinates (sin/cos)
    # This helps the AI understand that 23:00 is adjacent to 00:00
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

