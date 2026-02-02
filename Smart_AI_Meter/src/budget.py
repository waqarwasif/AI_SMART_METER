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
    # 1. ESTABLISH GOAL
    if target_bill_rs and target_bill_rs > 0:
        budget_units = 0
        money = target_bill_rs
        # Reverse Slab Logic
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
    else:
        if predicted_kwh > 300:
            target_units = 300
        elif predicted_kwh > 200:
            target_units = 200
        else:
            target_units = predicted_kwh * 0.9
        mode = "Efficiency Optimization"

    # 2. CALCULATE GAP
    projected_gap = predicted_kwh - target_units

    # Calculate safe daily limit
    safe_daily_limit = target_units / days_left if days_left > 0 else 0

    plan_data = {
        "mode": mode,
        "target_units": round(target_units, 1),
        "predicted_units": round(predicted_kwh, 1),
        "gap_units": round(projected_gap, 1),
        "daily_limit": round(safe_daily_limit, 1),
        "status": "SAFE",
        "actions": [],
    }

    if projected_gap <= 0:
        plan_data["message"] = "✅ You are under budget! Great job."
        plan_data["actions"].append("Maintain current usage patterns.")
        plan_data["actions"].append(
            "💡 **Tip:** Check 'Solar ROI' to see how to eliminate your bill."
        )

        # FIX: Ensure compatibility with legacy Tab 2
        plan_data["action_plan"] = plan_data["actions"]
        return plan_data

    plan_data["status"] = "WARNING"
    plan_data["message"] = f"⚠️ You are over target by {projected_gap:.1f} units."

    # 3. SOLVER
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
        active_devices = [
            {"name": "AC", "kw": 1.5, "priority": 1},
            {"name": "Heater", "kw": 2.0, "priority": 1},
        ]

    active_devices.sort(key=lambda x: x["kw"], reverse=True)
    daily_gap = projected_gap / 7
    solved_gap = 0

    for dev in active_devices:
        if solved_gap >= daily_gap:
            break
        if dev["type"] == "essential":
            continue

        needed = daily_gap - solved_gap
        hours_to_cut = needed / dev["kw"]

        if hours_to_cut > 4:
            hours_to_cut = 4
        if hours_to_cut < 0.3:
            hours_to_cut = 0.5

        saved = hours_to_cut * dev["kw"]
        solved_gap += saved

        hours = int(hours_to_cut)
        mins = int((hours_to_cut - hours) * 60)
        time_str = f"{hours} hr {mins} min" if hours > 0 else f"{mins} min"

        plan_data["actions"].append(
            f"Cut **{dev['name']}** usage by **{time_str}/day** (Saves {saved*7:.1f} kWh/week)"
        )

    plan_data["action_plan"] = plan_data["actions"]

    return plan_data



def calculate_cost_from_units(units):
    """
    Forward Slab Logic: Converts Units (kWh) -> Money (PKR).
    Used by app.py to display accurate projected bill.
    """
    cost = 0
    remaining_units = units

    # Slab 1: First 200 units @ 18
    if remaining_units > 200:
        cost += 200 * 18
        remaining_units -= 200

        # Slab 2: Next 100 units @ 30
        if remaining_units > 100:
            cost += 100 * 30
            remaining_units -= 100

            # Slab 3: Above 300 units @ 40
            cost += remaining_units * 40
        else:
            cost += remaining_units * 30
    else:
        cost += remaining_units * 18

    return cost
        