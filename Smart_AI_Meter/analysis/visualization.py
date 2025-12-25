import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Ensure the output directory exists
os.makedirs('graphs', exist_ok=True)

# Configuration for Aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
PEAK_START = 7   # 7 AM
PEAK_END = 19    # 7 PM

def save_plot(filename):
    """Helper to save plots consistently"""
    plt.tight_layout()
    plt.savefig(f'graphs/{filename}', dpi=300)
    plt.close()

# ==========================================
# PART 1: VISUALIZATIONS FOR CLEAN DATA (Historical)
# ==========================================

def plot_clean_daily_profile(df):
    """
    Graph 1: Daily Load Profile Consistency
    y-axis: Usage kWh, x-axis: Hour (0-23)
    Shows the average curve with a confidence interval (consistency).
    """
    plt.figure(figsize=(10, 6))
    # Lineplot automatically calculates mean and shadow (confidence interval)
    sns.lineplot(data=df, x='hour', y='usage_kwh', color='#1f77b4', linewidth=3)
    
    plt.title('Daily Load Profile Consistency', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Average Power Usage (kWh)', fontsize=12)
    plt.xticks(range(0, 24, 2))
    save_plot('1_clean_daily_profile.png')

def plot_clean_peak_distribution(df):
    """
    Graph 2: Peak vs Off-Peak Distribution (Donut Chart)
    """
    # Create a temporary column for categorization
    df['period'] = df['hour'].apply(
        lambda x: 'Peak (7AM-7PM)' if PEAK_START <= x < PEAK_END else 'Off-Peak'
    )
    usage_by_period = df.groupby('period')['usage_kwh'].sum()
    
    plt.figure(figsize=(8, 8))
    # Donut chart structure
    plt.pie(usage_by_period, labels=usage_by_period.index, autopct='%1.1f%%', 
            startangle=90, colors=['#aec7e8', '#1f77b4'], 
            pctdistance=0.85, wedgeprops=dict(width=0.3, edgecolor='white'))
    
    plt.title('Power Usage Distribution', fontsize=16, fontweight='bold')
    save_plot('2_clean_peak_distribution.png')

def plot_clean_full_pattern(df):
    """
    Graph 3: Power Usage vs Time (Full Month Pattern)
    x-axis: Total Hours, y-axis: Usage
    """
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['usage_kwh'], color='#2ca02c', linewidth=1.5, alpha=0.9)
    
    plt.title('Full Month Power Usage Pattern', fontsize=16, fontweight='bold')
    plt.xlabel('Time (Cumulative Hours)', fontsize=12)
    plt.ylabel('Usage (kWh)', fontsize=12)
    save_plot('3_clean_full_pattern.png')

def plot_clean_temp_correlation(df):
    """
    Graph 4: Temperature vs Usage (Dual Axis)
    x-axis: Hour (0-23), Left Y: Usage, Right Y: Temp
    """
    # Calculate average hourly profile
    avg_df = df.groupby('hour')[['usage_kwh', 'temperature_c']].mean().reset_index()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot Usage (Left Axis)
    color = 'tab:blue'
    ax1.set_xlabel('Hour of Day (0-23)', fontsize=12)
    ax1.set_ylabel('Avg Usage (kWh)', color=color, fontsize=12)
    ax1.plot(avg_df['hour'], avg_df['usage_kwh'], color=color, linewidth=3, label='Usage')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Plot Temperature (Right Axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Avg Temperature (°C)', color=color, fontsize=12)
    ax2.plot(avg_df['hour'], avg_df['temperature_c'], color=color, linestyle='--', linewidth=2, label='Temp')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Correlation: Temperature vs Energy Usage', fontsize=16, fontweight='bold')
    plt.xticks(range(0, 24, 2))
    save_plot('4_clean_temp_correlation.png')

def plot_clean_heatmap(df):
    """
    Graph 5: Usage Heatmap (Blurred/Smooth) - CORRECTED
    y-axis: Day of Month (Top to Bottom), x-axis: Hour
    """
    # Pivot to create the grid (Matrix form)
    pivot_table = df.pivot_table(index='day_of_month', columns='hour', values='usage_kwh', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    
    # CHANGED: origin='upper' puts Day 1 at the top
    plt.imshow(pivot_table, aspect='auto', cmap='magma', interpolation='bicubic', origin='upper')
    
    plt.colorbar(label='kWh Usage Intensity')
    plt.title('Heatmap: Usage Intensity (Day vs Hour)', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Month', fontsize=12)
    plt.xticks(range(0, 24, 2))
    
    # Adjust Y-ticks to show day numbers correctly
    if len(pivot_table) > 0:
         plt.yticks(range(0, len(pivot_table), 2), list(pivot_table.index)[::2])
         
    save_plot('5_clean_heatmap.png')


# ==========================================
# PART 2: VISUALIZATIONS FOR PREDICTED DATA (AI Output)
# ==========================================

def plot_pred_daily_profile(df):
    """
    Graph 6: Predicted Daily Profile (7-Day Average)
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='hour', y='predicted_usage_kwh', color='#ff7f0e', linewidth=3)
    
    plt.title('AI Forecast: Average Daily Profile (Next 7 Days)', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Predicted Usage (kWh)', fontsize=12)
    plt.xticks(range(0, 24, 2))
    save_plot('6_pred_daily_profile.png')

def plot_pred_peak_distribution(df):
    """
    Graph 7: Predicted Peak vs Off-Peak (Donut)
    """
    df['period'] = df['hour'].apply(
        lambda x: 'Peak (7AM-7PM)' if PEAK_START <= x < PEAK_END else 'Off-Peak'
    )
    usage_by_period = df.groupby('period')['predicted_usage_kwh'].sum()
    
    plt.figure(figsize=(8, 8))
    plt.pie(usage_by_period, labels=usage_by_period.index, autopct='%1.1f%%', 
            startangle=90, colors=['#ffbb78', '#ff7f0e'], 
            pctdistance=0.85, wedgeprops=dict(width=0.3, edgecolor='white'))
    
    plt.title('AI Forecast: Usage Distribution', fontsize=16, fontweight='bold')
    save_plot('7_pred_peak_distribution.png')

def plot_pred_full_forecast(df):
    """
    Graph 8: Predicted Usage vs Time (Full 7 Days)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['predicted_usage_kwh'], color='#d62728', linewidth=2, linestyle='-')
    
    plt.title('AI Forecast: 7-Day Usage Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Hours into Future', fontsize=12)
    plt.ylabel('Predicted Usage (kWh)', fontsize=12)
    save_plot('8_pred_full_forecast.png')

def plot_pred_temp_forecast(df):
    """
    Graph 9: Predicted Temp vs Usage (Full 7 Days Dual Axis)
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:orange'
    ax1.set_xlabel('Hours into Future', fontsize=12)
    ax1.set_ylabel('Pred Usage (kWh)', color=color, fontsize=12)
    ax1.plot(df.index, df['predicted_usage_kwh'], color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Forecast Temp (°C)', color=color, fontsize=12)
    ax2.plot(df.index, df['temperature_c'], color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('AI Forecast: Usage vs Temperature Context', fontsize=16, fontweight='bold')
    save_plot('9_pred_temp_forecast.png')

def plot_pred_heatmap(df):
    """
    Graph 10: Predicted Heatmap (Day vs Hour) - CORRECTED
    """
    # Create day index helper (0-23 is Day 1, etc.)
    if 'day_index' not in df.columns:
        # Assuming sequential hourly data
        df['day_index'] = (df.index // 24) + 1
        
    pivot_table = df.pivot_table(index='day_index', columns='hour', values='predicted_usage_kwh', aggfunc='mean')
    
    plt.figure(figsize=(10, 6))
    
    # CHANGED: origin='upper' puts Day 1 at the top
    plt.imshow(pivot_table, aspect='auto', cmap='inferno', interpolation='bicubic', origin='upper')
    
    plt.colorbar(label='Predicted kWh')
    plt.title('AI Forecast: Usage Intensity Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Forecast Day (1-7)', fontsize=12)
    plt.xticks(range(0, 24, 2))
    
    # Set Y-ticks to show Day 1, Day 2, etc.
    plt.yticks(range(0, 7), labels=[f'Day {i}' for i in range(1, 8)])
    
    save_plot('10_pred_heatmap.png')