
# AI Smart Meter Behavioral Advisor
## Overview
This project is a modular AI-driven Energy Advisor built in Python using Streamlit. The system is divided into logical modules so that the team can work independently on data processing, AI modeling, and UI design.

It uses automated data cleaning and Scikit-Learn (Isolation Forest) to detect hidden energy waste in Pakistani households.

## Project Phases
### Phase 1
### Objective: Project Infrastructure Tasks:

1) Initialize Git repository and connect to GitHub via terminal.

2) Setup .gitignore to protect .venv and private configurations.

3) Create basic Streamlit dashboard entry point (app.py).

4) Define the project's data structure and requirements.

### Phase 2
### Objective: Data Ingestion & "The Cleaner" Functions to Implement:

clean_data(): Read raw CSV files and handle missing NaN values and remove data-type errors.

Storage: data/raw_usage.csv

### Phase 3
### Objective: The "Pattern Finder" (Baseline Logic) Functions to Implement:

calculate_baseload(): Identify "Vampire Loads" (min usage between 2 AM - 5 AM).

group_usage(): Calculate average daily and hourly routines.

find_active_usage(): Subtract baseload from total consumption.

### Phase 4
Objective: The "AI Brain" (Anomaly Detection) Algorithm: IsolationForest Tasks:

Train model on the user's typical usage pattern.

Identify and flag unusual usage spikes as anomalies (-1).

Export anomaly results for user feedback.

### Phase 5
Objective: Smart Advisor & Slab Alerts Tasks:

Integrate Gemini API to generate a custom 7-day energy-saving plan.

Implement Pakistani Electricity Slab math to predict bill jumps.

Add professional visualization charts (Plotly/Seaborn).

## Directory Structure

```text
AI_SMART_METER/
├── Smart_AI_Meter/
    ├── src/                    # Logic Implementation Modules
    │   ├── __init__.py         # Package initializer
    │   ├── processor.py        # Data cleaning logic
    │   ├── predictor.py        # Training Model
    │   ├── forecaster.py       # Predictions
    │   ├── weather_service.py  # Fetching weather 
    │   └── recommender.py      # Hugging Face API
    ├── app.py                  # Main Entry point
    ├── .streamlit/             # Config & Secrets
    ├── data/                   # Persistent CSV Storage
        ├── prediction.csv      # Predicted data
        ├── Raw.csv             # Raw data
        └── processed.csv       # Processed output
    └── analysis/               # Developer Documentation
        └── visualization.py
 ```
## Modular Development Rules
1) Each module (e.g., processor.py) has its own independent logic.
Functions must include Docstrings explaining inputs and outputs.

2) Each team member must document their specific logic inside the helper/ folder.
Example: helper/ai_notes.txt.

3) Include: Dependencies used, logic thresholds, and testing notes.

4) Data Handling:
Use Pandas for all CSV manipulations.
Never modify the sample_data.csv directly; always export to a new variable or file.

5) Testing & Integration:
Test each script individually (e.g., python src/processor.py) before importing into app.py.
Only import the required functions into the main dashboard to keep code clean.
