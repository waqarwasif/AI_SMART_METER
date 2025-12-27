import pandas as pd
from huggingface_hub import InferenceClient
import time
import datetime

def get_ai_energy_plan(past_df, future_df, api_key, household_profile={}):
    """
    Generates Context-Aware Energy Plan using a HYBRID approach:
    - Python handles all logic, math, and season detection
    - AI only writes natural language based on pre-computed facts
    This ensures correctness while maintaining readability.
    """
    print("🤖 AI CONSULTANT: Analyzing Data & Building Strategy...")

    try:
        # --- STEP 1: DATA PREPARATION ---
        df_history = past_df.copy()
        if 'timestamp' not in df_history.columns:
            df_history = df_history.reset_index()
            for col in ['index', 'Datetime', df_history.columns[0]]:
                if col in df_history.columns:
                    df_history = df_history.rename(columns={col: 'timestamp'})
                    break
        
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], errors='coerce')

        # --- STEP 2: FORECAST ANALYSIS ---
        future_df_copy = future_df.copy()
        future_df_copy['date'] = pd.to_datetime(future_df_copy['timestamp']).dt.date
        future_df_copy['day_name'] = pd.to_datetime(future_df_copy['timestamp']).dt.day_name()
        
        daily_forecast = future_df_copy.groupby(['date', 'day_name']).agg({
            'predicted_usage_kwh': 'sum',
            'temperature_c': 'mean'
        }).reset_index()
        
        max_day = daily_forecast.loc[daily_forecast['predicted_usage_kwh'].idxmax()]
        min_day = daily_forecast.loc[daily_forecast['predicted_usage_kwh'].idxmin()]
        
        total_future_usage = daily_forecast['predicted_usage_kwh'].sum()
        avg_daily = total_future_usage / 7
        avg_temp = future_df['temperature_c'].mean()
        
        # Past usage calculation
        last_7_days = df_history.tail(168).copy()
        usage_col = next((c for c in ['usage_kwh', 'Usage', 'usage', 'kWh'] if c in last_7_days.columns), None)
        past_usage = last_7_days[usage_col].sum() if usage_col else 0
        
        # Calculate weekly average and project to monthly
        # We have 7 days of past data + 7 days of forecast = 14 days total
        total_14_days = past_usage + total_future_usage
        avg_daily_consumption = total_14_days / 14
        
        # Project to full 30-day month
        projected_monthly = avg_daily_consumption * 30
        
        # Billing tier calculation (based on monthly consumption)
        target_reduction = 0
        cost_per_unit = 16
        tier_name = "Standard"
        
        if projected_monthly > 700:
            # Calculate how much to reduce from monthly projection
            monthly_reduction_needed = projected_monthly - 700
            # Convert to 7-day target (what can be achieved in next week)
            target_reduction = int((monthly_reduction_needed / 30) * 7)
            tier_name = "CRITICAL"
            cost_per_unit = 42
        elif projected_monthly > 300:
            monthly_reduction_needed = projected_monthly - 300
            target_reduction = int((monthly_reduction_needed / 30) * 7)
            tier_name = "HIGH"
            cost_per_unit = 27
        elif projected_monthly > 200:
            monthly_reduction_needed = projected_monthly - 200
            target_reduction = int((monthly_reduction_needed / 30) * 7)
            tier_name = "WARNING"
            cost_per_unit = 22

        # Cap target reduction at 30% of weekly forecast (realistic limit)
        max_realistic_reduction = int(total_future_usage * 0.3)
        if target_reduction > max_realistic_reduction:
            target_reduction = max_realistic_reduction

        potential_savings = target_reduction * cost_per_unit if target_reduction > 0 else 0

        # --- STEP 3: SEASON DETECTION (Data-Driven) ---
        if avg_temp >= 32:
            actual_season = "Summer"
            season_type = "cooling"
        elif avg_temp >= 28:
            actual_season = "Spring"
            season_type = "moderate"
        elif avg_temp >= 24:
            actual_season = "Autumn"
            season_type = "moderate"
        else:
            actual_season = "Winter"
            season_type = "heating"
        
        print(f"🌡️ Detected Season: {actual_season} (avg temp: {avg_temp:.1f}°C)")

        # --- STEP 4: DEVICE FILTERING & CALCULATION ---
        residents = household_profile.get('residents', 4)
        selected_devices = household_profile.get('devices', [])
        
        # Device database with season relevance
        device_specs = {
            "Air Conditioner (AC)": {
                "watts": 1500, 
                "typical_hours": 8, 
                "seasons": ["Summer", "Spring"],
                "peak_times": "2:00 PM to 10:00 PM",
                "optimized_hours": "11:00 PM to 6:00 AM",
                "temp_rule": "Only use when temperature exceeds 28°C"
            },
            "Electric Heater": {
                "watts": 2000, 
                "typical_hours": 6, 
                "seasons": ["Winter", "Autumn"],
                "peak_times": "6:00 PM to 10:00 PM",
                "optimized_hours": "10:00 PM to 6:00 AM",
                "temp_rule": "Only use when temperature drops below 20°C"
            },
            "Electric Geyser": {
                "watts": 2000, 
                "typical_hours": 2, 
                "seasons": ["Winter", "Autumn", "Spring"],
                "peak_times": "6:00 AM to 8:00 AM, 6:00 PM to 9:00 PM",
                "optimized_hours": "Set to 50°C instead of 60°C",
                "temp_rule": "Reduce temperature setting during warmer days"
            },
            "Water Pump (Motor)": {
                "watts": 750, 
                "typical_hours": 2, 
                "seasons": ["All"],
                "peak_times": "Morning and evening",
                "optimized_hours": "6:00 AM to 7:00 AM, 10:00 PM to 11:00 PM",
                "temp_rule": "Schedule during off-peak electricity hours"
            },
            "Iron": {
                "watts": 1000, 
                "typical_hours": 1, 
                "seasons": ["All"],
                "peak_times": "Morning hours",
                "optimized_hours": "2:00 PM to 4:00 PM",
                "temp_rule": "Batch ironing once per week to reduce heating cycles"
            },
            "Washing Machine": {
                "watts": 500, 
                "typical_hours": 1.5, 
                "seasons": ["All"],
                "peak_times": "Morning hours",
                "optimized_hours": "2:00 PM to 4:00 PM",
                "temp_rule": "Use cold water cycles when possible"
            },
            "Microwave": {
                "watts": 1200, 
                "typical_hours": 0.5, 
                "seasons": ["All"],
                "peak_times": "Meal times",
                "optimized_hours": "Use for reheating only",
                "temp_rule": "Prefer microwave over oven for efficiency"
            },
            "EV Charger": {
                "watts": 7000, 
                "typical_hours": 3, 
                "seasons": ["All"],
                "peak_times": "Evening hours",
                "optimized_hours": "11:00 PM to 5:00 AM (off-peak rates)",
                "temp_rule": "Always charge overnight for maximum savings"
            }
        }
        
        # Filter to season-relevant devices
        relevant_devices = []
        filtered_out = []
        
        for device in selected_devices:
            if device in device_specs:
                device_seasons = device_specs[device]["seasons"]
                if "All" in device_seasons or actual_season in device_seasons:
                    relevant_devices.append(device)
                else:
                    filtered_out.append(device)
        
        # Calculate realistic savings breakdown
        device_breakdown = []
        total_calculated_savings = 0
        
        if target_reduction > 0 and relevant_devices:
            # Distribute savings proportionally based on device power
            total_device_watts = sum(device_specs[d]["watts"] * device_specs[d]["typical_hours"] 
                                    for d in relevant_devices)
            
            for device in relevant_devices:
                spec = device_specs[device]
                device_share = (spec["watts"] * spec["typical_hours"]) / total_device_watts
                device_saving = round(target_reduction * device_share, 1)
                
                # Don't let any single device exceed its theoretical maximum
                max_possible = (spec["watts"] * spec["typical_hours"] * 7) / 1000 * 0.4  # 40% reduction max
                device_saving = min(device_saving, max_possible)
                
                device_breakdown.append({
                    "name": device,
                    "saving_kwh": device_saving,
                    "daily_saving": round(device_saving / 7, 1),
                    "optimized_schedule": spec["optimized_hours"],
                    "rule": spec["temp_rule"]
                })
                total_calculated_savings += device_saving
        
        # Adjust if total doesn't match target (due to rounding)
        if total_calculated_savings > 0 and abs(total_calculated_savings - target_reduction) > 2:
            adjustment_factor = target_reduction / total_calculated_savings
            for item in device_breakdown:
                item["saving_kwh"] = round(item["saving_kwh"] * adjustment_factor, 1)
                item["daily_saving"] = round(item["saving_kwh"] / 7, 1)

        # --- STEP 5: DAY-BY-DAY PLANNING ---
        # Assign heavy tasks to low-usage days and create varied daily recommendations
        daily_plan = []
        sorted_days = daily_forecast.sort_values('predicted_usage_kwh')
        
        # Build list of specific tasks based on available devices
        heavy_tasks = []
        if "Iron" in relevant_devices:
            heavy_tasks.append("Complete all weekly ironing in one session")
        if "Washing Machine" in relevant_devices:
            heavy_tasks.append("Do major laundry loads (bed sheets, towels)")
        
        # Assign heavy tasks to lowest-usage days
        task_assigned_days = []
        for i, task in enumerate(heavy_tasks):
            if i < len(sorted_days):
                day = sorted_days.iloc[i]['day_name']
                task_assigned_days.append((day, task))
        
        # Generate unique recommendations for each day
        for idx, row in daily_forecast.iterrows():
            day = row['day_name']
            temp = row['temperature_c']
            usage = row['predicted_usage_kwh']
            
            # Check if this day has a heavy task assignment
            assigned_task = next((task for d, task in task_assigned_days if d == day), None)
            
            # If no heavy task, create device-specific recommendations
            if not assigned_task:
                # Create varied daily tasks based on relevant devices and usage level
                if usage < avg_daily:
                    # Low usage day - good for optional tasks
                    if "Water Pump (Motor)" in relevant_devices:
                        assigned_task = "Run water pump for tank filling during off-peak hours"
                    elif "Washing Machine" in relevant_devices:
                        assigned_task = "Good day for washing smaller loads"
                    else:
                        assigned_task = "Minimize high-power device usage today"
                else:
                    # Higher usage day - focus on efficiency
                    if season_type == "heating" and "Electric Geyser" in relevant_devices:
                        assigned_task = "Limit geyser usage to morning and evening hours only"
                    elif season_type == "cooling" and "Air Conditioner (AC)" in relevant_devices:
                        assigned_task = "Maximize natural ventilation, use AC sparingly"
                    else:
                        assigned_task = "Focus on turning off standby devices and unused lights"
            
            # Temperature-based advice (more specific and varied)
            if season_type == "heating":
                if temp < 18:
                    temp_advice = f"Very cold ({temp:.1f}°C) - Heater needed but limit to occupied rooms only"
                elif temp < 20:
                    temp_advice = f"Cold day ({temp:.1f}°C) - Use heater strategically in evening only"
                elif temp < 22:
                    temp_advice = f"Cool weather ({temp:.1f}°C) - Layer clothing, reduce heater by 2-3 hours"
                elif temp > 24:
                    temp_advice = f"Warmer day ({temp:.1f}°C) - No heating needed, open windows for ventilation"
                else:
                    temp_advice = f"Moderate temperature ({temp:.1f}°C) - Reduce heater usage by 2 hours"
            elif season_type == "cooling":
                if temp > 36:
                    temp_advice = f"Extreme heat ({temp:.1f}°C) - AC essential but set to 26°C, seal windows"
                elif temp > 33:
                    temp_advice = f"Very hot ({temp:.1f}°C) - Use AC in main room only, close unused areas"
                elif temp > 30:
                    temp_advice = f"Hot day ({temp:.1f}°C) - Limit AC to nighttime, use fans during day"
                elif temp < 28:
                    temp_advice = f"Cooler day ({temp:.1f}°C) - Skip AC entirely, ceiling fans sufficient"
                else:
                    temp_advice = f"Warm ({temp:.1f}°C) - Fans recommended, AC only if absolutely needed"
            else:
                if temp > 26:
                    temp_advice = f"Warm weather ({temp:.1f}°C) - Natural ventilation with fans"
                elif temp < 22:
                    temp_advice = f"Pleasant cool day ({temp:.1f}°C) - Open windows, no HVAC needed"
                else:
                    temp_advice = f"Ideal temperature ({temp:.1f}°C) - Natural ventilation sufficient"
            
            daily_plan.append({
                "day": day,
                "task": assigned_task,
                "temp_advice": temp_advice
            })

        # --- STEP 6: BUILD STRUCTURED FACTS FOR AI ---
        facts_for_ai = f"""
HOUSEHOLD FACTS:
- Residents: {residents} people
- Detected Season: {actual_season} (average temperature: {avg_temp:.1f}°C)
- Season Type: {season_type} season (focus on {"cooling" if season_type == "cooling" else "heating" if season_type == "heating" else "balanced energy use"})
- Current Billing Tier: {tier_name}
- Past 7 days actual usage: {past_usage:.1f} kWh
- Next 7 days predicted usage: {total_future_usage:.1f} kWh
- Average daily consumption: {avg_daily_consumption:.1f} kWh
- Projected monthly usage: {projected_monthly:.1f} kWh
- Reduction Target for next 7 days: {target_reduction} kWh (this is {target_reduction/7:.2f} kWh per day)
- Potential Savings: Rs. {potential_savings:,.0f}

RELEVANT DEVICES FOR THIS SEASON:
{chr(10).join(f"- {d}" for d in relevant_devices) if relevant_devices else "- Only basic appliances (lights, fans, refrigerator)"}

FILTERED OUT (Not relevant for {actual_season}):
{chr(10).join(f"- {d}" for d in filtered_out) if filtered_out else "- None"}

DEVICE SAVINGS BREAKDOWN (totals {sum(d['saving_kwh'] for d in device_breakdown):.1f} kWh over 7 days):
{chr(10).join(f"- {d['name']}: {d['saving_kwh']:.1f} kWh total ({d['daily_saving']:.1f} kWh/day) - Schedule: {d['optimized_schedule']}" for d in device_breakdown) if device_breakdown else "- Focus on standby power reduction and lighting efficiency"}

DAILY FORECAST WITH RECOMMENDATIONS:
{chr(10).join(f"- {row['day_name']}: {row['predicted_usage_kwh']:.1f} kWh predicted at {row['temperature_c']:.1f}°C" for _, row in daily_forecast.iterrows())}

DAY-BY-DAY ACTION PLAN:
{chr(10).join(f"- {p['day']}: {p['task']} | {p['temp_advice']}" for p in daily_plan)}

SEASON-SPECIFIC OPTIMIZATION ({actual_season}):
{"These tips are for WINTER/HEATING season:" if season_type == "heating" else "These tips are for SUMMER/COOLING season:" if season_type == "cooling" else "These tips are for MODERATE season:"}
{chr(10).join(f"- {d['name']}: {d['rule']}" for d in device_breakdown) if device_breakdown else "- Use LED bulbs, unplug chargers, optimize refrigerator settings"}
"""

        # --- STEP 7: SIMPLE PROMPT FOR AI (Just Format the Facts) ---
        prompt = f"""You are writing an Energy Audit Report based on verified calculations.

All the numbers, device lists, and recommendations below have been pre-calculated and verified. Your job is to present this information clearly in a professional report format.

{facts_for_ai}

Write a report with these EXACT sections:

**SECTION 1: TARGET BREAKDOWN**
Present the device savings breakdown shown above. Make sure the individual savings add up to {target_reduction} kWh total. Use the exact numbers provided.

**SECTION 2: DEVICE-SPECIFIC DAILY SCHEDULE**
For each relevant device listed above, explain its optimized schedule and daily savings. Use the exact schedules and numbers provided.

**SECTION 3: DAY-BY-DAY ACTION PLAN**
Present the 7-day plan shown above. Include the task assignment and temperature advice for each day.

**SECTION 4: {actual_season} OPTIMIZATION**
List 3-5 specific tips relevant to {actual_season} season. Base these on the season-specific optimizations shown above. Every tip must include a number (kWh or Rs. or temperature setting).

CRITICAL RULES:
1. Use ONLY the devices listed as "RELEVANT DEVICES" above
2. Do NOT mention any devices in "FILTERED OUT" list
3. All numbers must exactly match the calculations provided
4. The season is {actual_season} with {season_type} focus - all advice must match this
5. Keep explanations clear and actionable

Write the report now."""

        # --- STEP 8: API CALL ---
        free_chat_models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta"
        ]
        
        client = InferenceClient(token=api_key)

        for model_name in free_chat_models:
            try:
                print(f"📡 Connecting to {model_name.split('/')[-1]}...")
                response = client.chat_completion(
                    model=model_name,
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You are a report writer. Present the provided facts clearly without adding or changing any numbers. The season is {actual_season} with {season_type} focus."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.2,
                )
                
                if response and response.choices:
                    generated_text = response.choices[0].message.content
                    
                    # Basic validation
                    has_numbers = any(char.isdigit() for char in generated_text)
                    
                    if has_numbers:
                        # Add disclaimer if devices were filtered
                        if filtered_out:
                            disclaimer = f"\n📌 NOTE: The following devices were excluded from this report because they are not relevant for {actual_season}: {', '.join(filtered_out)}\n\n"
                            return disclaimer + generated_text
                        return generated_text
                    else:
                        print(f"⚠️ {model_name} gave invalid response, trying next...")
                        continue
                    
            except Exception as e:
                print(f"⚠️ {model_name} error: {str(e)[:100]}")
                continue

        # Fallback with pre-calculated facts
        fallback = f"""⌛ AI models are currently busy. Here's your energy plan based on verified calculations:

**YOUR SITUATION:**
- Season: {actual_season} ({avg_temp:.1f}°C average)
- Target: Reduce {target_reduction} kWh = {target_reduction/7:.2f} kWh per day
- Potential Savings: Rs. {potential_savings:,.0f}

**DEVICE OPTIMIZATION:**
{chr(10).join(f"- {d['name']}: Reduce by {d['daily_saving']:.1f} kWh/day using schedule: {d['optimized_schedule']}" for d in device_breakdown) if device_breakdown else "Focus on basic efficiency: LED bulbs, unplugging devices, optimizing refrigerator"}

**KEY ACTIONS:**
- Schedule heavy tasks (ironing, laundry) on your lowest-usage days: {', '.join([day['day'] for day in daily_plan[:3]])}
- {"Set heater timers for night hours only" if season_type == "heating" else "Run AC only during nighttime" if season_type == "cooling" else "Maximize natural ventilation"}

Try generating the full report again in a moment."""
        
        return fallback

    except Exception as e:
        return f"❌ System Error: {str(e)}"