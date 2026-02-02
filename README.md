![License](https://img.shields.io/badge/License-MIT-green.svg)

# AI Smart Meter Behavioral Advisor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aismartmeter-behavioraladvisor.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![AI Model](https://img.shields.io/badge/Model-Random%20Forest%20%2B%20Llama%203.2-orange)

**Live Demo:** [Launch Dashboard](https://aismartmeter-behavioraladvisor.streamlit.app/)

---

## Overview

The **AI Smart Meter Behavioral Advisor** is a modular, AI-driven energy consultancy platform designed to identify hidden energy waste and optimize household electricity budgets.

Unlike standard monitoring apps, this system acts as an intelligent "Energy Accountant." It combines **Random Forest Regression** for load forecasting, **Pakistani Tariff Slab Logic** for financial accuracy, and a **Generative AI Agent (Llama 3.2)** that uses RAG (Retrieval-Augmented Generation) principles to interpret complex energy data into actionable, human-readable advice.

---

## Key Features

### 1. AI & Machine Learning Core

* **Predictive Engine:** Utilizes Random Forest Regression to forecast energy usage for the next 7 days, training on historical data and cyclical time features (Sine/Cosine encoding).
* **Intelligent Agent (RAG):** A context-aware AI assistant integrated with the Hugging Face API. It retrieves the mathematical budget plan and generates professional audit reports in PDF format.

### 2. Financial Optimization

* **Reverse Budget Calculator:** Users input a target monthly budget (e.g., "Rs. 5,000"), and the system calculates a precise **Safe Daily Limit** (kWh), accounting for tiered slab rates.
* **Solar ROI Planner:** Determines the optimal solar system size (kW), total investment cost, and payback period based on the user's consumption profile and local sunlight data.

### 3. Real-Time Data Simulation

* **IoT Sensor Stream:** A dedicated module (`simulate_sensor.py`) mimics hardware smart meter behavior, generating real-time streams of Voltage, Current, and Power data.
* **Physics-Based Cleaning:** The processor module enforces physical constraints (e.g., max load limits) and handles missing data using linear interpolation to ensure model stability.

### 4. Environmental Integration

* **Live Weather Integration:** Fetches real-time hourly temperature forecasts for Karachi via the Open-Meteo API to refine load predictions (e.g., adjusting for AC usage during heatwaves).

---

## Directory Structure
```text
AI_SMART_METER/
├── .streamlit/                 # Configuration & Secrets
├── analysis/                   # Developer Visualization Tools
│   └── visualization.py        # Matplotlib/Seaborn plotting logic
├── data/                       # Data Storage
│   ├── raw/                    # Uploaded user datasets
│   └── live_stream.csv         # Generated IoT simulation data
├── graphs/                     # Exported charts for PDF reports
├── src/                        # Core Logic Modules
│   ├── __init__.py
│   ├── budget.py               # Reverse Budgeting & Slab Logic
│   ├── forecaster.py           # 7-Day Future Prediction Loop
│   ├── predictor.py            # Random Forest Training Engine
│   ├── processor.py            # Data Cleaning & Feature Engineering
│   ├── recommender.py          # AI Agent (Hugging Face / Llama)
│   ├── solar.py                # Solar System & ROI Calculator
│   └── weather_service.py      # Open-Meteo API Integration
├── app.py                      # Main Streamlit Dashboard Entry Point
├── requirements.txt            # Python Dependencies
├── simulate_sensor.py          # IoT Hardware Simulator
└── README.md                   # Project Documentation
```

---

## Installation & Usage

### Prerequisites

- Python 3.8 or higher
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/ai-smart-meter.git
cd ai-smart-meter
```

### Step 2: Install Dependencies

It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### Step 3: Generate Sensor Data

To simulate a live smart meter connection, run the simulator script in a separate terminal. This will create and update `data/live_stream.csv`.
```bash
python simulate_sensor.py
```

*(Let this run in the background for 10-20 seconds to generate initial data)*

### Step 4: Launch the Dashboard
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Application Modules

The dashboard is structured into four strategic tabs:

### Prediction & Strategy

- Upload CSV data (or use the simulated stream).
- Visualizes 7-day load forecasts and projected bill amounts.
- Generates the downloadable "AI Energy Audit" PDF.

### Reverse Budget

- Input a target budget (e.g., Rs. 3,000).
- The system reverses the tariff slab logic to provide a daily usage cap (e.g., "Limit usage to 3.5 kWh/day").

### Solar ROI

- Input an average monthly bill.
- Receives a tailored Solar System recommendation, including system size, cost, and a 10-year savings projection.

### AI Assistant

- A chatbot interface where users can ask specific questions about their energy plan. The AI has context on the user's specific predicted usage and budget gaps.

---

## Development Standards

- **Logic Isolation:** Each module in `src/` operates independently with clear inputs and outputs. `app.py` is strictly for UI rendering and state management.
- **Data Handling:** Pandas is used for all data manipulation. Physical constraints (e.g., max 20kW load) are enforced strictly in `processor.py` before any AI training occurs.
- **Testing:** Utility scripts (e.g., `src/solar.py`, `src/budget.py`) can be executed independently to verify logic before integration.

---

## Technologies Used

Developed with **Python**, **Scikit-Learn**, and **Streamlit**.
API used **HuggingFace**

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Connect with me: [https://linktr.ee/waqarwasif](https://linktr.ee/waqarwasif)