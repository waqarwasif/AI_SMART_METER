import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ---------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------
os.makedirs("graphs", exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.autolayout"] = False

FIG_STD = (5, 3)
FIG_WIDE = (6, 3)
FIG_HEAT = (6, 4)
DPI = 120

PEAK_START = 7
PEAK_END = 19


def save_plot(name):
    """
    Saves the current matplotlib figure to the graphs directory
    and closes it to free memory.
    """
    plt.tight_layout()
    plt.savefig(f"graphs/{name}", dpi=DPI, bbox_inches="tight")
    plt.close()


# ---------------------------------------------
# HISTORICAL DATA PLOTS
# ---------------------------------------------
def plot_clean_daily_profile(df):
    avg = df.groupby("hour")["usage_kwh"].mean()
    plt.figure(figsize=FIG_STD)
    plt.plot(avg.index, avg.values, color="#1f77b4", linewidth=2)
    plt.title("Average Daily Load Profile")
    plt.xlabel("Hour of Day")
    plt.ylabel("Avg Usage (kWh)")
    plt.xticks(range(0, 24, 3))
    save_plot("1_clean_daily_profile.png")


def plot_clean_peak_distribution(df):
    df = df.copy()
    df["period"] = df["hour"].apply(
        lambda h: "Peak" if PEAK_START <= h < PEAK_END else "Off-Peak"
    )
    usage = df.groupby("period")["usage_kwh"].sum()

    plt.figure(figsize=(4, 4))
    plt.pie(
        usage,
        labels=usage.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff7f0e", "#1f77b4"],
        wedgeprops=dict(width=0.35),
    )
    plt.title("Energy Usage Distribution")
    save_plot("2_clean_peak_distribution.png")


def plot_clean_full_pattern(df):
    plt.figure(figsize=FIG_WIDE)
    plt.plot(range(len(df)), df["usage_kwh"], linewidth=1, color="#2ca02c")
    plt.title("Full Usage Pattern (Relative Time)")
    plt.xlabel("Time Index")
    plt.ylabel("Usage (kWh)")
    save_plot("3_clean_full_pattern.png")


def plot_clean_temp_correlation(df):
    avg = df.groupby("hour")[["usage_kwh", "temperature_c"]].mean()

    fig, ax1 = plt.subplots(figsize=FIG_STD)
    ax1.plot(avg.index, avg["usage_kwh"], color="#1f77b4", linewidth=2)
    ax1.set_ylabel("Avg Usage (kWh)", color="#1f77b4")
    ax1.set_xlabel("Hour of Day")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(avg.index, avg["temperature_c"], "--", color="#d62728", linewidth=2)
    ax2.set_ylabel("Avg Temperature (°C)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("Temperature vs Energy Usage")
    plt.xticks(range(0, 24, 3))
    save_plot("4_clean_temp_correlation.png")


def plot_clean_heatmap(df):
    # Pivot table: Day of Month x Hour
    pivot = (
        df.pivot_table(
            index="day_of_month", columns="hour", values="usage_kwh", aggfunc="mean"
        )
        .sort_index()
        .fillna(0)
    )

    plt.figure(figsize=FIG_HEAT)
    sns.heatmap(
        pivot,
        cmap="magma",
        vmin=pivot.min().min(),
        vmax=pivot.max().max(),
        cbar_kws={"label": "kWh"},
    )
    plt.title("Usage Intensity Heatmap")
    plt.xlabel("Hour")
    plt.ylabel("Day of Month")
    save_plot("5_clean_heatmap.png")


# ---------------------------------------------
# PREDICTED DATA PLOTS
# ---------------------------------------------
def plot_pred_daily_profile(df):
    avg = df.groupby("hour")["predicted_usage_kwh"].mean()
    plt.figure(figsize=FIG_STD)
    plt.plot(avg.index, avg.values, color="#ff7f0e", linewidth=2)
    plt.title("Predicted Daily Profile")
    plt.xlabel("Hour")
    plt.ylabel("Predicted Usage (kWh)")
    plt.xticks(range(0, 24, 3))
    save_plot("6_pred_daily_profile.png")


def plot_pred_peak_distribution(df):
    df = df.copy()
    df["period"] = df["hour"].apply(
        lambda h: "Peak" if PEAK_START <= h < PEAK_END else "Off-Peak"
    )
    usage = df.groupby("period")["predicted_usage_kwh"].sum()

    plt.figure(figsize=(4, 4))
    plt.pie(
        usage,
        labels=usage.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ffbb78", "#ff7f0e"],
        wedgeprops=dict(width=0.35),
    )
    plt.title("Predicted Energy Distribution")
    save_plot("7_pred_peak_distribution.png")


def plot_pred_full_forecast(df):
    plt.figure(figsize=FIG_WIDE)
    plt.plot(range(len(df)), df["predicted_usage_kwh"], linewidth=1.5, color="#9467bd")
    plt.title("7-Day Forecast (Relative Time)")
    plt.xlabel("Forecast Hour Index")
    plt.ylabel("Predicted Usage (kWh)")
    save_plot("8_pred_full_forecast.png")


def plot_pred_temp_forecast(df):
    fig, ax1 = plt.subplots(figsize=FIG_WIDE)

    ax1.plot(range(len(df)), df["predicted_usage_kwh"], color="#ff7f0e", linewidth=2)
    ax1.set_ylabel("Predicted Usage (kWh)", color="#ff7f0e")
    ax1.tick_params(axis="y", labelcolor="#ff7f0e")

    ax2 = ax1.twinx()
    ax2.plot(range(len(df)), df["temperature_c"], "--", color="#d62728", linewidth=2)
    ax2.set_ylabel("Forecast Temperature (°C)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_xlabel("Forecast Hour Index")
    plt.title("Forecast: Usage vs Temperature")
    save_plot("9_pred_temp_forecast.png")


def plot_pred_heatmap(df):
    df = df.copy()

    # Ensure timestamp is datetime type for calculation
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Calculate a "Forecast Day" (Day 1, Day 2, etc.)
    df["forecast_day"] = (
        df["timestamp"].dt.date - df["timestamp"].dt.date.min()
    ).apply(lambda x: x.days + 1)

    # Pivot table: Forecast Day x Hour
    pivot = df.pivot_table(
        index="forecast_day",
        columns="hour",
        values="predicted_usage_kwh",
        aggfunc="mean",
    ).sort_index()

    # Scale colors robustly (ignoring extreme outliers for better visual contrast)
    vmin = pivot.stack().quantile(0.05)
    vmax = pivot.stack().quantile(0.95)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        pivot,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        linewidths=0.2,
        cbar_kws={"label": "Predicted Usage (kWh)"},
    )

    plt.title("Predicted Usage Heatmap (7-Day Forecast)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Forecast Day")
    save_plot("10_pred_heatmap.png")
