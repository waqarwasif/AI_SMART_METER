# src/solar.py
import pandas as pd
import numpy as np


def calculate_solar_roi(avg_monthly_bill):
    """
    Calculates Solar System Size, Cost, and ROI based on the user's bill.
    """
    # --- 1. CONSTANTS (Pakistani Market Rates) ---
    COST_PER_KW = 120000  # Avg cost of 1kW system (panels + inverter + structure)
    UNIT_RATE = 45  # Avg cost per unit (kWh) from grid (blended rate)
    UNITS_PER_KW_DAILY = 4.2  # Daily generation per 1kW (Karachi sunlight avg)

    # --- 2. SYSTEM SIZING LOGIC ---
    # Reverse calc: Bill -> Units needed
    avg_monthly_units = avg_monthly_bill / UNIT_RATE
    required_daily_units = avg_monthly_units / 30

    # We need enough panels to cover daily needs
    system_size_kw = np.ceil(required_daily_units / UNITS_PER_KW_DAILY)

    # --- 3. FINANCIAL PROJECTION LOGIC ---
    total_system_cost = system_size_kw * COST_PER_KW
    daily_production_units = system_size_kw * UNITS_PER_KW_DAILY
    monthly_savings = daily_production_units * 30 * UNIT_RATE
    yearly_savings = monthly_savings * 12

    # Break-even point: When do savings > cost?
    payback_years = total_system_cost / yearly_savings

    # --- 4. 10-YEAR PROFIT CHART ---
    years = list(range(11))
    # Cumulative savings over time
    cumulative_savings = [0] + [yearly_savings * i for i in range(1, 11)]
    # Initial cost stays flat (simplified)
    cumulative_cost = [total_system_cost] * 11

    # Net Profit = Savings - Initial Cost
    net_profit = [sav - cost for sav, cost in zip(cumulative_savings, cumulative_cost)]

    return {
        "system_size_kw": system_size_kw,
        "total_cost": total_system_cost,
        "monthly_savings": monthly_savings,
        "payback_years": payback_years,
        "chart_data": pd.DataFrame({"Year": years, "Net Profit (PKR)": net_profit}),
    }
