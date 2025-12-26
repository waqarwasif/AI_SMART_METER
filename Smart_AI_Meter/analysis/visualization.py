import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# ---------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------
os.makedirs("graphs", exist_ok=True)

plt.style.use("default")
sns.set_theme(style="whitegrid", context="paper")

# TRUE FINAL SIZE CONTROL
FIG_SIZE_STANDARD = (5, 3)     # SMALLER
FIG_SIZE_WIDE = (6, 3)
FIG_SIZE_HEATMAP = (6, 4)
DPI_SETTING = 120              # LOWER DPI = smaller PNG

PEAK_START = 7
PEAK_END = 19


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        f"graphs/{filename}",
        dpi=DPI_SETTING,
        bbox_inches="tight"
    )
    plt.close()


# ---------------------------------------------
# CLEAN DATA (HISTORICAL)
# ---------------------------------------------
def plot_clean_daily_profile(df):
    plt.figure(figsize=FIG_SIZE_STANDARD)
    sns.lineplot(data=df, x="hour", y="usage_kwh", linewidth=2)
    plt.title("Daily Load Profile")
    plt.xlabel("Hour")
    plt.ylabel("Usage (kWh)")
    plt.xticks(range(0, 24, 3))
    save_plot("1_clean_daily_profile.png")


def plot_clean_peak_distribution(df):
    plt.figure(figsize=(4, 4))
    df = df.copy()
    df["period"] = df["hour"].apply(
        lambda x: "Peak" if PEAK_START <= x < PEAK_END else "Off-Peak"
    )
    usage = df.groupby("period")["usage_kwh"].sum()

    plt.pie(
        usage,
        labels=usage.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.35)
    )
    plt.title("Peak vs Off-Peak Usage")
    save_plot("2_clean_peak_distribution.png")


def plot_clean_full_pattern(df):
    plt.figure(figsize=FIG_SIZE_WIDE)
    plt.plot(df.index, df["usage_kwh"], linewidth=1)
    plt.title("Full Usage Pattern")
    plt.xlabel("Time")
    plt.ylabel("kWh")
    save_plot("3_clean_full_pattern.png")


def plot_clean_temp_correlation(df):
    plt.figure(figsize=FIG_SIZE_STANDARD)
    avg = df.groupby("hour")[["usage_kwh", "temperature_c"]].mean()

    plt.plot(avg.index, avg["usage_kwh"], label="Usage")
    plt.plot(avg.index, avg["temperature_c"], linestyle="--", label="Temp")

    plt.legend()
    plt.title("Temp vs Usage")
    save_plot("4_clean_temp_correlation.png")


def plot_clean_heatmap(df):
    plt.figure(figsize=FIG_SIZE_HEATMAP)
    pivot = df.pivot_table(
        index="day_of_month",
        columns="hour",
        values="usage_kwh",
        aggfunc="mean"
    )
    sns.heatmap(pivot, cmap="magma")
    plt.title("Usage Heatmap")
    save_plot("5_clean_heatmap.png")


# ---------------------------------------------
# PREDICTED DATA (AI OUTPUT)
# ---------------------------------------------
def plot_pred_daily_profile(df):
    plt.figure(figsize=FIG_SIZE_STANDARD)
    sns.lineplot(data=df, x="hour", y="predicted_usage_kwh", linewidth=2)
    plt.title("Predicted Daily Profile")
    save_plot("6_pred_daily_profile.png")


def plot_pred_peak_distribution(df):
    plt.figure(figsize=(4, 4))
    df = df.copy()
    df["period"] = df["hour"].apply(
        lambda x: "Peak" if PEAK_START <= x < PEAK_END else "Off-Peak"
    )
    usage = df.groupby("period")["predicted_usage_kwh"].sum()

    plt.pie(
        usage,
        labels=usage.index,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.35)
    )
    plt.title("Predicted Distribution")
    save_plot("7_pred_peak_distribution.png")


def plot_pred_full_forecast(df):
    plt.figure(figsize=FIG_SIZE_WIDE)
    plt.plot(df["timestamp"], df["predicted_usage_kwh"], linewidth=1.5)
    plt.xticks(rotation=45)
    plt.title("7-Day Forecast")
    save_plot("8_pred_full_forecast.png")


def plot_pred_temp_forecast(df):
    plt.figure(figsize=FIG_SIZE_WIDE)
    plt.plot(df["timestamp"], df["predicted_usage_kwh"], label="Usage")
    plt.plot(df["timestamp"], df["temperature_c"], linestyle="--", label="Temp")
    plt.legend()
    plt.xticks(rotation=45)
    plt.title("Forecast: Usage vs Temp")
    save_plot("9_pred_temp_forecast.png")


def plot_pred_heatmap(df):
    df = df.copy()
    if "day_index" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        start = df["timestamp"].dt.date.iloc[0]
        df["day_index"] = df["timestamp"].apply(
            lambda x: (x.date() - start).days + 1
        )

    plt.figure(figsize=FIG_SIZE_HEATMAP)
    pivot = df.pivot_table(
        index="day_index",
        columns="hour",
        values="predicted_usage_kwh",
        aggfunc="mean"
    )
    sns.heatmap(pivot, cmap="inferno")
    plt.title("Predicted Heatmap")
    save_plot("10_pred_heatmap.png")
