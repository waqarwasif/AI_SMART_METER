# src/budget.py


def calculate_budget_plan(target_bill_rs, current_usage_kwh, days_left):
    """
    Reverse-engineers the electricity tariff to find safe daily usage limits.
    """
    # --- 1. INVERSE SLAB LOGIC (Simplified Pakistan Tariff) ---
    # We try to find how many units 'x' result in 'target_bill_rs'
    # Rate structure: First 200 @ Rs 18, Next 100 @ Rs 30, Above 300 @ Rs 40

    budget_units = 0
    remaining_money = target_bill_rs

    # Tier 1: First 200 units (Protected)
    cost_tier_1 = 200 * 18
    if remaining_money >= cost_tier_1:
        budget_units += 200
        remaining_money -= cost_tier_1

        # Tier 2: Next 100 units
        cost_tier_2 = 100 * 30
        if remaining_money >= cost_tier_2:
            budget_units += 100
            remaining_money -= cost_tier_2

            # Tier 3: Above 300
            budget_units += remaining_money / 40
        else:
            budget_units += remaining_money / 30
    else:
        budget_units += remaining_money / 18

    # --- 2. DAILY LIMIT CALCULATION ---
    remaining_units = budget_units - current_usage_kwh

    if remaining_units <= 0:
        return {
            "status": "CRITICAL",
            "daily_limit": 0,
            "message": "⚠️ STOP! You have already exceeded your budget.",
        }

    daily_limit = remaining_units / days_left

    # --- 3. DEVICE RECOMMENDATION LOGIC ---
    # What fits in this daily limit?
    advice = []

    # Baseline load (Fridge/Lights/Fans) is usually ~3-4 kWh/day
    if daily_limit < 4:
        advice.append("❌ CRITICAL: Budget too low. Run ONLY Fridge, Lights & Fans.")
        advice.append("⛔ DO NOT turn on AC, Iron, or Motor.")
    else:
        surplus = daily_limit - 4  # Energy left for heavy devices
        ac_hours = int(surplus / 1.5)  # 1.5 ton AC consumes ~1.5 kWh

        advice.append("✅ Basic home load (Fridge/Lights) is covered.")
        if ac_hours > 0:
            advice.append(f"❄️ You can run the AC for **{ac_hours} hours** tonight.")
        else:
            advice.append("⚠️ Budget tight: Avoid AC usage to stay on track.")

    return {
        "status": "SAFE" if daily_limit > 5 else "WARNING",
        "daily_limit": daily_limit,
        "message": "\n".join(advice),
    }
