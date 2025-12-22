# AI Smart Meter Behavioral Advisor
**Developed by 3Pearls**

An intelligent energy monitoring system designed to help Pakistani households reduce electricity bills using Machine Learning and Generative AI.

## Key Features
- **Data Ingestion:** Automatically cleans and standardizes messy smart meter CSV data.
- **Vampire Load Detection:** Identifies "always-on" appliances wasting energy.
- **AI Anomaly Detection:** Uses Scikit-Learn (Isolation Forest) to find unusual usage spikes.
- **Smart Advisor:** Generates a personalized 7-Day saving plan via Gemini API.
- **Slab Alert:** Real-time monitoring to prevent jumping into higher electricity tax brackets.

## Tech Stack
- **Frontend:** Streamlit
- **AI/ML:** Scikit-Learn, Google Gemini API
- **Data:** Pandas, Numpy
- **Visuals:** Plotly, Seaborn

## Project Structure
- `src/`: Core logic (Processing, AI, Recommender)
- `data/`: Sample CSV files
- `assets/`: UI styling and images
- `app.py`: Main dashboard entry point

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

Run the app:
   ```bash
streamlit run app.py