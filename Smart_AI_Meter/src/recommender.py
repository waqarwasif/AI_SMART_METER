import pandas as pd
from huggingface_hub import InferenceClient
import time
import datetime


def get_ai_energy_plan(
    past_df, future_df, api_key, household_profile={}, agent_plan=None
):
    """
    Generates Context-Aware Energy Plan using a HYBRID approach:
    - If 'agent_plan' is provided, the Report strictly follows the Agent's math.
    - If not, it falls back to standard estimation logic.
    - AI only writes natural language based on these pre-computed facts.
    """
    print("🤖 AI CONSULTANT: Analyzing Data & Building Strategy...")

    try:
        # --- STEP 1: DATA PREPARATION ---
        df_history = past_df.copy()
        if "timestamp" not in df_history.columns:
            df_history = df_history.reset_index()
            for col in ["index", "Datetime", df_history.columns[0]]:
                if col in df_history.columns:
                    df_history = df_history.rename(columns={col: "timestamp"})
                    break

        df_history["timestamp"] = pd.to_datetime(
            df_history["timestamp"], errors="coerce"
        )

        # --- STEP 2: FORECAST ANALYSIS (Context for the Report) ---
        future_df_copy = future_df.copy()
        future_df_copy["date"] = pd.to_datetime(future_df_copy["timestamp"]).dt.date
        future_df_copy["day_name"] = pd.to_datetime(
            future_df_copy["timestamp"]
        ).dt.day_name()

        daily_forecast = (
            future_df_copy.groupby(["date", "day_name"])
            .agg({"predicted_usage_kwh": "sum", "temperature_c": "mean"})
            .reset_index()
        )

        total_future_usage = daily_forecast["predicted_usage_kwh"].sum()
        avg_daily = total_future_usage / 7
        avg_temp = future_df["temperature_c"].mean()

        # Past usage calculation
        last_7_days = df_history.tail(168).copy()
        usage_col = next(
            (
                c
                for c in ["usage_kwh", "Usage", "usage", "kWh"]
                if c in last_7_days.columns
            ),
            None,
        )
        past_usage = last_7_days[usage_col].sum() if usage_col else 0

        # Project to monthly
        total_14_days = past_usage + total_future_usage
        avg_daily_consumption = total_14_days / 14
        projected_monthly = avg_daily_consumption * 30

        # --- STEP 3: SEASON DETECTION ---
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

        # --- STEP 4: AGENT VS LEGACY LOGIC ---

        agent_instructions = ""
        target_reduction = 0
        potential_savings = 0
        tier_name = "Standard"

        # [A] THE INTELLIGENT AGENT PATH (Priority)
        if agent_plan:
            target_reduction = agent_plan.get("gap_units", 0)
            tier_name = agent_plan.get("status", "Calculated")

            # Create a strict instruction block for the AI
            agent_instructions = f"""
*** 🚨 CALCULATED AGENT PLAN (MUST FOLLOW STRICTLY) ***
The Energy Accountant Agent has solved the user's budget equation. 
You must output these EXACT actions. Do not invent new ones.

- User Mode: {agent_plan.get('mode', 'Optimization')}
- Prediction: {agent_plan.get('predicted_units', 0)} kWh
- Target Goal: {agent_plan.get('target_units', 0)} kWh
- GAP TO CLOSE: {agent_plan.get('gap_units', 0)} kWh

REQUIRED ACTIONS (Write these exactly in the Action Plan):
{chr(10).join(f"- {action}" for action in agent_plan.get('actions', []))}
"""

        # [B] THE LEGACY ESTIMATION PATH (Fallback)
        else:
            # Old Billing Logic
            cost_per_unit = 16
            if projected_monthly > 700:
                target_reduction = int(((projected_monthly - 700) / 30) * 7)
                tier_name = "CRITICAL"
                cost_per_unit = 42
            elif projected_monthly > 300:
                target_reduction = int(((projected_monthly - 300) / 30) * 7)
                tier_name = "HIGH"
                cost_per_unit = 27
            elif projected_monthly > 200:
                target_reduction = int(((projected_monthly - 200) / 30) * 7)
                tier_name = "WARNING"
                cost_per_unit = 22

            # Cap reduction
            max_realistic = int(total_future_usage * 0.3)
            if target_reduction > max_realistic:
                target_reduction = max_realistic

            potential_savings = target_reduction * cost_per_unit

            agent_instructions = f"""
*** ESTIMATED PLAN (No Agent Data) ***
- Target Reduction: {target_reduction} kWh
- Estimated Savings: Rs. {potential_savings}
- Tier: {tier_name}
(Generate generic device advice based on season)
"""

        # --- STEP 5: DEVICE CONTEXT (For Flavor Text) ---
        residents = household_profile.get("residents", 4)
        selected_devices = household_profile.get("devices", [])

        # --- STEP 6: BUILD FACTS FOR AI ---
        facts_for_ai = f"""
CONTEXTUAL FACTS:
- Residents: {residents}
- Season: {actual_season} ({avg_temp:.1f}°C) -> Focus on {season_type}
- Past 7 Days Usage: {past_usage:.1f} kWh
- Next 7 Days Forecast: {total_future_usage:.1f} kWh
- Projected Monthly: {projected_monthly:.1f} kWh
- Billing Tier: {tier_name}

{agent_instructions}

AVAILABLE DEVICES IN HOME:
{', '.join(selected_devices) if selected_devices else "Standard Basic Appliances"}

DAILY WEATHER FORECAST (Use for day-by-day advice):
{chr(10).join(f"- {row['day_name']}: {row['predicted_usage_kwh']:.1f} kWh predicted at {row['temperature_c']:.1f}°C" for _, row in daily_forecast.iterrows())}
"""

        # --- STEP 7: PROMPT CONSTRUCTION ---
        prompt = f"""You are writing an Energy Audit Report based on strict mathematical calculations.

{facts_for_ai}

Write a professional report with these EXACT sections:

**SECTION 1: THE SITUATION**
Summarize the user's current status (Forecast vs Target). If an Agent Plan exists, state the "Gap to Close".

**SECTION 2: REQUIRED ACTIONS (THE PLAN)**
If "CALCULATED AGENT PLAN" is provided above, list those exact actions. 
If not, recommend general reductions based on the Season and Available Devices.

**SECTION 3: 7-DAY WEATHER STRATEGY**
Look at the Daily Weather Forecast above. Give a specific tip for each day based on the temperature (e.g., "Monday is hot, use fans").

**SECTION 4: {actual_season.upper()} SEASON TIPS**
Give 3 short, specific technical tips for {season_type} efficiency.

CRITICAL RULES:
1. If the "CALCULATED AGENT PLAN" is present, YOU MUST use those numbers. Do not hallucinate different numbers.
2. Keep it professional and concise.
3. Address the user directly.

Write the report now."""

        # --- STEP 8: API CALL ---
        free_chat_models = [
            "meta-llama/Llama-3.2-3B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
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
                            "content": "You are a precise Energy Auditor. You follow the Agent's calculations exactly.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1500,
                    temperature=0.2,  # Low temp for factual accuracy
                )

                if response and response.choices:
                    generated_text = response.choices[0].message.content
                    if any(char.isdigit() for char in generated_text):
                        return generated_text

            except Exception as e:
                print(f"⚠️ {model_name} error: {str(e)[:100]}")
                continue

        # Fallback if AI fails
        return f"""**⚠️ AI Connection Busy - Here is your Raw Plan:**

**TARGET:** {agent_plan.get('target_units', 'N/A') if agent_plan else 'N/A'} kWh
**PREDICTED:** {agent_plan.get('predicted_units', 'N/A') if agent_plan else total_future_usage:.1f} kWh

**REQUIRED ACTIONS:**
{chr(10).join(f"- {action}" for action in agent_plan.get('actions', [])) if agent_plan else "Focus on reducing AC and Heater usage."}

*(Please try generating the full report again later)*"""

    except Exception as e:
        return f"❌ System Error: {str(e)}"
