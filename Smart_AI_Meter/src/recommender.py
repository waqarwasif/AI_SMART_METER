import pandas as pd
from huggingface_hub import InferenceClient
import time

def get_ai_energy_plan(past_df, future_df, api_key, household_profile={}):
    """
    Uses Hugging Face InferenceClient with chat_completion (FREE).
    Works with models that support the chat completion endpoint.
    """
    print("🤖 AI CONSULTANT: Analyzing your energy data...")

    try:
        # --- STEP 1: DATA PREP ---
        df_history = past_df.copy()
        
        if 'timestamp' not in df_history.columns:
            df_history = df_history.reset_index()
            if 'index' in df_history.columns:
                df_history = df_history.rename(columns={'index': 'timestamp'})
            elif 'Datetime' in df_history.columns:
                df_history = df_history.rename(columns={'Datetime': 'timestamp'})
            if 'timestamp' not in df_history.columns:
                df_history = df_history.rename(columns={df_history.columns[0]: 'timestamp'})
        
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], errors='coerce')

        # Validate columns
        if 'predicted_usage_kwh' not in future_df.columns:
            return "❌ ERROR: 'predicted_usage_kwh' column missing."
        if 'temperature_c' not in future_df.columns:
            return "❌ ERROR: 'temperature_c' column missing."

        # --- STEP 2: CALCULATIONS ---
        future_df_copy = future_df.copy()
        future_df_copy['date'] = pd.to_datetime(future_df_copy['timestamp']).dt.date
        
        daily_forecast = future_df_copy.groupby('date').agg({
            'predicted_usage_kwh': 'sum',
            'temperature_c': 'mean'
        }).reset_index()
        
        total_future_usage = daily_forecast['predicted_usage_kwh'].sum()
        
        # Get past usage
        last_7_days = df_history.tail(168).copy()
        last_7_days['date'] = last_7_days['timestamp'].dt.date
        
        usage_col = None
        for col in ['usage_kwh', 'Usage', 'usage', 'kWh']:
            if col in last_7_days.columns:
                usage_col = col
                break
        
        past_usage = last_7_days.groupby('date')[usage_col].sum().sum() if usage_col else 0
        projected_total = past_usage + total_future_usage
        
        # Pakistani Slab Logic
        slab_warning = "✅ Safe"
        units_to_cut = 0
        
        if projected_total > 700:
            units_to_cut = int(projected_total - 700)
            slab_warning = f"🚨 CRITICAL: {projected_total:.0f} units at Rs.30/unit"
        elif projected_total > 300:
            units_to_cut = int(projected_total - 300)
            slab_warning = f"⚠️ HIGH: {projected_total:.0f} units at Rs.24/unit"
        elif projected_total > 200:
            units_to_cut = int(projected_total - 200)
            slab_warning = f"⚠️ WARNING: {projected_total:.0f} units at Rs.20/unit"

        # Context
        residents = household_profile.get('residents', 4)
        devices = household_profile.get('devices', [])
        avg_temp = daily_forecast['temperature_c'].mean()
        
        # Device tips
        tips = []
        if any("AC" in d for d in devices):
            tips.append(f"AC running at {avg_temp:.0f}°C average - set thermostat to 26°C to save 6-8% per degree")
        if any("Pump" in d or "Motor" in d for d in devices):
            tips.append("Water Pump consumes 1.5-2kW/hour - shift operation to off-peak 6-8AM")
        if any("Geyser" in d for d in devices):
            tips.append("Electric Geyser uses ~2kW/hour - limit heating to 30 mins max per day")
        if any("Iron" in d for d in devices):
            tips.append("Iron uses 1-1.5kW - batch ironing sessions instead of daily use")

        # Forecast summary (compact)
        forecast_text = []
        for _, row in daily_forecast.iterrows():
            day = pd.to_datetime(row['date']).strftime("%a")
            forecast_text.append(f"{day}:{row['predicted_usage_kwh']:.1f}kWh@{row['temperature_c']:.0f}C")

        # --- STEP 3: COMPACT PROMPT FOR CHAT COMPLETION ---
        prompt = f"""Energy Advisor Task:

Household: {residents} people
Status: {slab_warning}
Must Cut: {units_to_cut} units
Devices: {', '.join(devices[:4]) if devices else 'Standard'}
Forecast: {' | '.join(forecast_text)}

Instructions:
1. List 3 specific actions to cut {units_to_cut} units
2. Identify hottest day and give cooling advice
3. Be technical and brief (max 150 words)
4. Focus on device-specific savings, not generic tips"""

        # --- STEP 4: USE HUGGINGFACE INFERENCE CLIENT (FREE) ---
        # InferenceClient supports chat_completion for FREE models
        free_chat_models = [
            "meta-llama/Llama-3.2-3B-Instruct",  # Small, fast, FREE
            "HuggingFaceH4/zephyr-7b-beta",      # Reliable backup
            "mistralai/Mistral-7B-Instruct-v0.2" # Last resort
        ]
        
        last_error = "Unknown"

        for model_name in free_chat_models:
            try:
                print(f"📡 Connecting to {model_name.split('/')[-1]}...")
                
                # Create client
                client = InferenceClient(token=api_key)
                
                # Call chat_completion (FREE endpoint)
                response = client.chat_completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a professional energy consultant. Be brief and technical."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                # Extract text
                if response and response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content
                    if text and len(text) > 30:
                        print(f"✅ Success!")
                        return text
                
                last_error = f"{model_name.split('/')[-1]}: Empty response"
                continue
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Model loading
                if '503' in error_msg or 'loading' in error_msg:
                    last_error = f"{model_name.split('/')[-1]} is loading"
                    print(f"⏳ Loading... trying next")
                    time.sleep(2)
                    continue
                
                # Unauthorized
                elif '401' in error_msg or 'unauthorized' in error_msg:
                    return "❌ Invalid API Key. Get one at: https://huggingface.co/settings/tokens"
                
                # Rate limit
                elif '429' in error_msg or 'rate' in error_msg:
                    last_error = "Rate limited"
                    print(f"⚠️ Rate limited")
                    time.sleep(3)
                    continue
                
                # Model not found or deprecated
                elif '404' in error_msg or '410' in error_msg:
                    last_error = f"{model_name.split('/')[-1]} unavailable (404/410)"
                    print(f"❌ Model deprecated/unavailable")
                    continue
                
                # Other errors
                else:
                    last_error = f"{type(e).__name__}: {str(e)[:80]}"
                    print(f"⚠️ Error: {error_msg[:50]}")
                    continue

        # FALLBACK - Return calculated analysis
        return f"""❌ AI Models Temporarily Unavailable

━━━━━━━━━━━━━━━━━━━━━━━━
📊 CALCULATED ENERGY PLAN
━━━━━━━━━━━━━━━━━━━━━━━━

Status: {slab_warning}
7-Day Forecast: {total_future_usage:.1f} kWh
Target Cut: {units_to_cut} units to avoid higher slab

IMMEDIATE ACTIONS:
{chr(10).join(f'• {tip}' for tip in tips) if tips else '• Shift high-load appliances to off-peak hours (11PM-6AM)'}
• Unplug standby devices (TVs, chargers) - saves 5-10 units/month
• Use natural ventilation when temperature < 28°C

HOTTEST DAY: {daily_forecast.loc[daily_forecast['temperature_c'].idxmax(), 'date'].strftime('%A')} ({daily_forecast['temperature_c'].max():.0f}°C)
Strategy: Pre-cool home before peak hours, use fans, keep curtains closed

━━━━━━━━━━━━━━━━━━━━━━━━
Error: {last_error}
Note: AI service busy. Retry in 2 minutes for personalized plan."""

    except Exception as e:
        return f"❌ SYSTEM ERROR: {type(e).__name__} - {str(e)[:150]}"