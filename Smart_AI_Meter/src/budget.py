# src/budget.py
import math

DEVICE_LIBRARY = {
    "Air Conditioner (1.5 Ton)": {"kw": 1.5, "priority": 1, "type": "cooling"},
    "Air Conditioner (1 Ton)": {"kw": 1.0, "priority": 1, "type": "cooling"},
    "Inverter AC": {"kw": 0.8, "priority": 1, "type": "cooling"},
    "Electric Heater": {"kw": 2.0, "priority": 1, "type": "heating"},
    "Iron": {"kw": 1.0, "priority": 3, "type": "utility"},
    "Water Motor (1 HP)": {"kw": 0.75, "priority": 2, "type": "utility"},
    "Washing Machine": {"kw": 0.5, "priority": 3, "type": "utility"},
    "Microwave": {"kw": 1.2, "priority": 4, "type": "kitchen"},
    "Electric Geyser": {"kw": 2.0, "priority": 2, "type": "heating"},
    "Gaming PC": {"kw": 0.4, "priority": 4, "type": "entertainment"},
    "Refrigerator": {"kw": 0.15, "priority": 1, "type": "essential"},
    "Deep Freezer": {"kw": 0.2, "priority": 1, "type": "essential"},
    "Ceiling Fan": {"kw": 0.08, "priority": 5, "type": "basic"},
    "LED Lights (10 Bulbs)": {"kw": 0.12, "priority": 5, "type": "basic"},
}


def calculate_budget_plan(
    target_bill_rs=None,
    current_usage_kwh=0,
    days_left=30,
    user_selected_devices=None,
    predicted_kwh=0,
):
    # --- 1. CONVERT MONEY -> UNITS (Inverse Slab) ---
    budget_units = 0
    if target_bill_rs and target_bill_rs > 0:
        money = target_bill_rs
        if money >= (200 * 18):
            budget_units += 200
            money -= 200 * 18
            if money >= (100 * 30):
                budget_units += 100
                money -= 100 * 30
                budget_units += money / 40
            else:
                budget_units += money / 30
        else:
            budget_units += money / 18
        target_units = int(budget_units)
        mode = "Strict Budget"
            budget_units += money / 18
        target_units = int(budget_units)
        mode = "Strict Budget"
    else:
        # Default Logic if no budget given
        if predicted_kwh > 0:
            target_units = predicted_kwh * 0.9  # Aim for 10% saving
        else:
            target_units = 300  # Fallback default
        mode = "Efficiency Optimization"

    # --- 2. DETERMINE STATUS & GAP ---
    plan_data = {
        "mode": mode,
        "target_units": round(target_units, 1),
        "predicted_units": round(predicted_kwh, 1),
        "actions": [],
        "status": "SAFE",
    }

    # LOGIC BRANCH: Are we predicting (Tab 1) or just calculating (Tab 2)?
    if predicted_kwh > 0:
        # --- TAB 1: PREDICTION MODE ---
        projected_gap = predicted_kwh - target_units
        plan_data["gap_units"] = round(projected_gap, 1)
        # Safe Daily Limit based on TARGET
        safe_daily_limit = target_units / days_left if days_left > 0 else 0
        plan_data["daily_limit"] = round(safe_daily_limit, 1)

        if projected_gap <= 0:
            plan_data["message"] = (
                "✅ **You are safe!** Your target budget covers your predicted usage."
            )
            plan_data["actions"].append("Maintain current usage patterns.")
            plan_data["action_plan"] = plan_data["actions"]
            return plan_data
        else:
            plan_data["status"] = "WARNING"
            plan_data["message"] = (
                f"⚠️ **Action Needed!** You are projected to exceed your budget by {projected_gap:.1f} kWh."
            )
            gap_to_solve = projected_gap  # We need to cut this amount

    else:
        # --- TAB 2: CALCULATOR MODE ---
        # We just want to know: "How much can I use per day?"
        # We subtract what we ALREADY used (current_usage_kwh)
        remaining_units = target_units - current_usage_kwh
        safe_daily_limit = remaining_units / days_left if days_left > 0 else 0

        plan_data["gap_units"] = 0
        plan_data["daily_limit"] = round(safe_daily_limit, 1)

        if remaining_units <= 0:
            plan_data["status"] = "CRITICAL"
            plan_data["message"] = (
                f"🚨 **Budget Exceeded!** You have already used {current_usage_kwh} units. Stop usage immediately."
            )
            gap_to_solve = 0
        else:
            plan_data["status"] = "SAFE"
            plan_data["message"] = (
                f"✅ **Budget Plan:** You can use **{safe_daily_limit:.1f} kWh/day** for the next {days_left} days."
            )
            # In calculator mode, we 'solve' for the whole daily limit to show what fits
            gap_to_solve = 0

    # --- 3. SOLVER (Generate Device Advice) ---
    # We generate advice based on the Daily Limit (what fits?)
    # OR the Gap (what to cut?) depending on the mode.

    active_devices = []
    if user_selected_devices:
        for dev_name in user_selected_devices:
            match = next(
                (v for k, v in DEVICE_LIBRARY.items() if dev_name.lower() in k.lower()),
                None,
            )
            if match:
                active_devices.append({"name": dev_name, **match})

    if not active_devices:
        active_devices = [{"name": "AC", "kw": 1.5}, {"name": "Heater", "kw": 2.0}]

    if plan_data["status"] == "WARNING":
        # Strategy: Suggest CUTS to close the gap
        active_devices.sort(key=lambda x: x["kw"], reverse=True)
        daily_cut_needed = gap_to_solve / 7  # Spread over a week
        solved = 0
        for dev in active_devices:
            if solved >= daily_cut_needed:
                break
            hours = (daily_cut_needed - solved) / dev["kw"]
            if hours > 4:
                hours = 4
            if hours < 0.3:
                hours = 0.5

            h = int(hours)
            m = int((hours - h) * 60)
            plan_data["actions"].append(
                f"Cut **{dev['name']}** by **{h}hr {m}min/day**"
            )
            solved += hours * dev["kw"]

    elif plan_data["status"] == "SAFE" and predicted_kwh == 0:
        # Strategy: Suggest WHAT FITS in the daily limit (For Tab 2)
        base_load = 4  # Fridge + Lights
        surplus = safe_daily_limit - base_load
        if surplus < 0:
            plan_data["actions"].append(
                "⚠️ Limit is very low! Run ONLY Fridge & Lights."
            )
        else:
            for dev in active_devices:
                if dev["type"] == "essential":
                    continue
                hours = surplus / dev["kw"]
                if hours >= 1:
                    plan_data["actions"].append(
                        f"🟢 You can run **{dev['name']}** for **{int(hours)} hours/day**"
                    )

    # Compatibility Copy
    plan_data["action_plan"] = plan_data["actions"]
    return plan_data


def calculate_cost_from_units(units):
    """Forward Slab Logic"""
    cost = 0
    rem = units
    if rem > 200:
        cost += 200 * 18
        rem -= 200
        if rem > 100:
            cost += 100 * 30
            rem -= 100
            cost += rem * 40
        else:
            cost += rem * 30
    else:
        cost += rem * 18
    return cost
