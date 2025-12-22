
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

load_data(): Read raw CSV files.

clean_data(): Handle missing NaN values and remove data-type errors.

standardize_data(): Ensure hourly timestamps are aligned for computation.

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

# AI_SMART_METER/
│
├── src/ # Logic Implementation Modules
│ ├── __init__.py # Package initializer
│ ├── processor.py # Data cleaning logic
│ ├── analytics.py # Baseload & pattern math
│ ├── ml_logic.py # AI/Anomaly detection
│ └── recommender.py # Gemini API & Advice logic
│
├── Smart_AI_Meter/ # Main Dashboard Files
│ ├── app.py # Main Entry point
│ ├── .streamlit/ # Config & Secrets
│ └── assets/ # UI Styling (CSS)
│
├── data/ # Persistent CSV Storage
│ ├── raw_usage.csv # Messy input data
│ └── processed_data.csv # Computable output
│
└── helper/ # Developer Documentation
├── data_cleaning.txt
├── ai_model_notes.txt
└── api_integration.txt

## Modular Development Rules
Each module (e.g., processor.py) has its own independent logic.

Functions must include Docstrings explaining inputs and outputs.

Each team member must document their specific logic inside the helper/ folder.

Example: helper/ai_notes.txt.

Include: Dependencies used, logic thresholds, and testing notes.

Data Handling:

Use Pandas for all CSV manipulations.

Never modify the raw_usage.csv directly; always export to a new variable or file.

Testing & Integration:

Test each script individually (e.g., python src/processor.py) before importing into app.py.

Only import the required functions into the main dashboard to keep code clean.