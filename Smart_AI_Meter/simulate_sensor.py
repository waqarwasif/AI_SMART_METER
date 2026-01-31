import pandas as pd
import numpy as np
import time
import os

FILE_PATH = "data/live_stream.csv"
os.makedirs("data", exist_ok=True)

# Initialize file with headers if it doesn't exist
if not os.path.exists(FILE_PATH):
    pd.DataFrame(columns=["timestamp", "voltage", "current", "power_kw"]).to_csv(
        FILE_PATH, index=False
    )

print("📡 IoT Sensor Simulation Started... (Data is streaming to data/live_stream.csv)")
print("Press Ctrl+C to stop.")

while True:
    
    now = pd.Timestamp.now()
    voltage = np.random.normal(220, 2)  # 220V +/- 2V fluctuation
    current = np.random.normal(8, 3)  # 8 Amps +/- 3A load change

    # 2. Calculate Power (P = V * I)
    
    if np.random.random() > 0.8:
        current += 10  # Sudden load spike

    power = (voltage * current) / 1000  # Convert Watts to kW

    # 3. Append to CSV
    new_row = pd.DataFrame(
        [[now, voltage, current, power]],
        columns=["timestamp", "voltage", "current", "power_kw"],
    )

    new_row.to_csv(FILE_PATH, mode="a", header=False, index=False)

    print(f"⚡ Reading: {power:.2f} kW | {voltage:.1f} V")
    time.sleep(2)  # 2-second sampling rate
