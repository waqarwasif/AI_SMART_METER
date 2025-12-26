import pandas as pd
from huggingface_hub import InferenceClient
import time
import datetime

def get_season(month):
    """Returns the season in Karachi based on the month."""
    if month in [11, 12, 1, 2]:
        return "Winter (Heating/Geyser Season)"
    elif month in [3, 4]:
        return "Spring (Moderate/Transition)"
    elif month in [5, 6, 7, 8, 9]:
        return "Summer (Peak AC Season)"
    else:
        return "Autumn (Transition)"

def get_ai_energy_plan(past_df, future_df, api_key, household_profile={}):
    """
    Generates a Context-Aware Energy Plan.
    - Knows the specific Month/Season in Karachi.
    - Strictly restricts advice to USER-SELECTED devices only.
    """
    print("🤖 AI CONSULTANT: Analyzing Season & Devices...")

    try:
        # --- STEP 1: DATA PREP ---
        df_history = past_df.copy()
        if 'timestamp' not in df_history.columns:
            df_history = df_history.reset_index()
            # Try finding the timestamp column
            found_col = False
            for col in ['index', 'Datetime', df_history.columns[0]]:
                if col in df_history.columns:
                    df_history = df_history.rename(columns={col: 'timestamp'})
                    found_col = True
                    break
        
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], errors='coerce')

        # --- STEP 2: SEASON & DATE CONTEXT ---
        # Get the first date from the forecast to determine current time
        future_df_copy = future_df.copy()
        future_df_copy['date'] = pd.to_datetime(future_df_copy['timestamp']).dt.date
        
        current_date = future_df_copy['date'].min()
        current_month = current_date.month
        current_season = get_season(current_month)
        formatted_date = current_date.strftime("%d %B %Y")

        # --- STEP 3: CALCULATIONS ---
        daily_forecast = future_df_copy.groupby('date').agg({
            'predicted_usage_kwh': 'sum',
            'temperature_c': 'mean'
        }).reset_index()
        
        total_future_usage = daily_forecast['predicted_usage_kwh'].sum()
        
        # Estimate Past Usage
        last_7_days = df_history.tail(168).copy()
        usage_col = next((c for c in ['usage_kwh', 'Usage', 'usage', 'kWh'] if c in last_7_days.columns), None)
        past_usage = last_7_days[usage_col].sum() if usage_col else 0
        
        projected_total = past_usage + total_future_usage
        
        # Slab Logic
        slab_warning = "✅ Standard Tier (Safe)"
        units_to_cut = 0
        if projected_total > 700:
            units_to_cut = int(projected_total - 700)
            slab_warning = f"🚨 CRITICAL: Projected {projected_total:.0f} units (Rs. 42/unit tier)"
        elif projected_total > 300:
            units_to_cut = int(projected_total - 300)
            slab_warning = f"⚠️ HIGH: Projected {projected_total:.0f} units (Rs. 27/unit tier)"
        elif projected_total > 200:
            units_to_cut = int(projected_total - 200)
            slab_warning = f"⚠️ WARNING: Approaching 200 Units (Rs. 22/unit tier)"

        # --- STEP 4: STRICT DEVICE CONTEXT ---
        residents = household_profile.get('residents', 4)
        selected_devices = household_profile.get('devices', [])
        
        if not selected_devices:
            device_str = "Standard Lights & Fans only (No heavy appliances selected)"
            device_constraint = "Focus ONLY on lighting, fans, and vampire loads."
        else:
            device_str = ', '.join(selected_devices)
            device_constraint = f"STRICTLY limit your advice to these devices: {device_str}. Do NOT give advice for ACs, Geysers, or Heaters if they are not listed above."

        # Forecast Block
        forecast_lines = []
        for _, row in daily_forecast.iterrows():
            d_name = pd.to_datetime(row['date']).strftime("%A")
            forecast_lines.append(f"- {d_name}: {row['predicted_usage_kwh']:.1f} kWh | {row['temperature_c']:.1f}°C")
        forecast_block = "\n".join(forecast_lines)

        # --- STEP 5: PROMPT ENGINEERING (Season & Device Strictness) ---
        prompt = f"""You are a Senior Energy Auditor for a household in Karachi, Pakistan but Don't mention that in the report.
        
**CURRENT CONTEXT:**
- **Date:** {formatted_date}
- **Season:** {current_season} use the weather for generating advice accordingly.
- **Projected Usage:** {projected_total:.1f} kWh over next 7 days.
- **Residents:** {residents} these are the number of people in the house affecting usage, consider this in your advice.
- **Billing Status:** {slab_warning} (Target Cut: {units_to_cut} .. mention this in the report)

**ACTIVE DEVICE LIST (STRICT):**
{device_str} these are the ONLY high usage devices you can give advice on.

**7-DAY FORECAST:**
{forecast_block} these are the predicted daily usages and temperatures for the next week. I want you to generate a day-by-day plan based on this.

**INSTRUCTIONS:**
Write a "Weekly Energy Strategy Report" with these specific rules:
1. **Context Awareness:** Since it is {current_season}, tailor your advice accordingly (e.g., if Winter, focus on Geysers/Heaters; if Summer, focus on AC/Fans).
2. **Device Strictness:** {device_constraint}
3. **Reduction Strategy:** specific plan to save {units_to_cut} units using ONLY the active devices.
4. **Clarity & Actionability:** Use simple language, bullet points, and clear daily tasks.


**REPORT SECTIONS:**
1. Executive Strategy: How to hit the reduction target this week.
2. Operational Schedule (Day-by-Day): Assign chores (Laundry/Ironing) to specific days based on the forecast.
3. Seasonal Recommendations: Specific tips for {current_season} in Karachi.
"""

        # --- STEP 6: API CALL ---
        free_chat_models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "HuggingFaceH4/zephyr-7b-beta",
            "mistralai/Mistral-7B-Instruct-v0.2"
        ]
        
        client = InferenceClient(token=api_key)

        for model_name in free_chat_models:
            try:
                print(f"📡 Connecting to {model_name.split('/')[-1]}...")
                response = client.chat_completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise Energy Consultant. Follow device constraints strictly."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1200,
                    temperature=0.5  # Lower temperature for stricter adherence
                )
                if response and response.choices:
                    return response.choices[0].message.content
                    
            except Exception as e:
                print(f"⚠️ {model_name} error: {str(e)[:50]}")
                continue

        return "❌ AI Busy. Please try again."

    except Exception as e:
        return f"❌ System Error: {str(e)}"