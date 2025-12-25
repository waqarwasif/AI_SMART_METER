import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_model(df):
    """
    The AI Engine: 
    1. Selects the best features (Math + Physics).
    2. Validates accuracy on the last 20% of data (to prove it works).
    3. Retrains on 100% of data (to be ready for the future).
    """
    print("🧠 Starting AI Training Sequence...")

    # 1. Feature Selection Strategy
    feature_candidates = [
        'temperature_c',   # The Physics (Weather)
        'is_weekend',      # The Logic (Behavior)
        'week_of_month',   # The Bill Cycle
        'hour_sin', 'hour_cos', # The Clock (Cyclical)
        'day_sin', 'day_cos',   # The Week (Cyclical)
        'month_sin', 'month_cos' # The Season (Cyclical)
    ]
    
    # CRITICAL: We only select features that actually exist in your dataframe.
    available_features = [f for f in feature_candidates if f in df.columns]
    
    target_col = 'usage_kwh'
    
    # Safety Check
    if not available_features:
        raise ValueError("❌ No valid features found for training!")

    print(f"   👉 Training on {len(available_features)} Features: {available_features}")

    # 2. The Validation Phase (The "Mock Exam")
    # We split the data: 80% Past (Train) vs 20% Future (Test)
    split_point = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    
    # Train a temporary model just for testing
    model_test = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model_test.fit(train_df[available_features], train_df[target_col])
    
    # Generate Accuracy Report
    predictions = model_test.predict(test_df[available_features])
    mae = mean_absolute_error(test_df[target_col], predictions)
    
    # MAPE calculation (Mean Absolute Percentage Error)
    # Added small epsilon (0.001) to avoid division by zero
    mape = np.mean(np.abs((test_df[target_col] - predictions) / (test_df[target_col] + 0.001))) * 100
    accuracy = 100 - mape
    
    print("\n" + "="*40)
    print("       MODEL PERFORMANCE REPORT       ")
    print("="*40)
    print(f"✅ Validation Accuracy:   {accuracy:.2f}%")
    print(f"⚠️ Margin of Error:       {mape:.2f}%")
    print(f"📉 Mean Absolute Error:   {mae:.4f} kWh")
    print("="*40 + "\n")

    # 3. The Production Phase (The "Final Exam")
    print("🚀 Retraining on 100% of Data for Deployment...")
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    final_model.fit(df[available_features], df[target_col])
    
    print("✅ AI Model Ready.")

    # Package the metrics so the UI can display them
    metrics = {
        "accuracy": accuracy,
        "mape": mape,
        "mae": mae
    }

    # Return 3 items: Model, Features, AND Metrics
    return final_model, available_features, metrics