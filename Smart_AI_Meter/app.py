import streamlit as st
import pandas as pd
import os
import time

# --- IMPORT YOUR MODULES ---
from src.processor import clean_data
from src.predictor import train_model
from src.forecaster import predict_next_week
# Added Recommender Import
from src.recommender import get_ai_energy_plan

# Import the visualization module
try:
    from analysis import visualization as viz
except ImportError:
    st.error("⚠️ Could not import 'visualization'. Please ensure 'visualization.py' is in the 'analysis' folder.")

# ---------------------------------------------
# 1. Page Configuration
# ---------------------------------------------
st.set_page_config(
    page_title="⚡ AI Smart Energy Advisor",
    layout="wide"
)

# ---------------------------------------------
# 2. UI: Hero Section & Styling
# ---------------------------------------------
st.markdown("""
<div style="background-color:#0b132b; padding:20px; border-radius:10px; margin-bottom: 20px;">
    <h1 style="color:#00f5d4; margin:0;">⚡ AI Smart Energy Advisor</h1>
    <p style="color:#e0e1dd; margin:0;">Turning Energy Numbers into Human Action</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Custom CSS
st.markdown("""
<style>
.stButton > button {
    background: linear-gradient(90deg, #00f5d4, #4cc9f0);
    color: #0b132b;
    border-radius: 10px;
    padding: 0.8rem 2rem;
    font-weight: bold;
    border: none;
    font-size: 16px; 
    width: 100%;
}
.stButton > button:hover {
    box-shadow: 0 0 20px #00f5d4;
    transform: scale(1.02);
}

/* Sub-box styling for accuracy metrics */
.sub-metric-box {
    padding: 8px 15px;
    border-radius: 8px;
    text-align: center;
    font-size: 14px;
    font-weight: 600;
    margin-top: 5px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 3. Logic: The Full Pipeline
# ---------------------------------------------
uploaded_file = st.file_uploader("📂 Upload electricity usage CSV", type=["csv"], key="upload_ui")

# Initialize session state
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'household_profile' not in st.session_state:
    st.session_state['household_profile'] = {}
if 'df_clean' not in st.session_state:
    st.session_state['df_clean'] = None

if uploaded_file:
    # --- STEP 1: SAVE RAW FILE ---
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("graphs", exist_ok=True) 

    raw_file_path = os.path.join("data/raw", uploaded_file.name)
    with open(raw_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ Raw file received: {uploaded_file.name}")

    # --- NEW SECTION: HOUSEHOLD PROFILE ---
    st.markdown("### 🏠 Household Context")
    st.info("Please tell us about your home so the AI can generate a smarter plan.")
    
    col_profile_1, col_profile_2 = st.columns(2)
    
    with col_profile_1:
        num_people = st.number_input("👨‍👩‍👧‍👦 Number of Residents", min_value=1, max_value=20, value=4)
    
    with col_profile_2:
        # Multiselect is cleaner than many checkboxes
        heavy_devices = st.multiselect(
            "🔌 Select High-Load Devices you own:",
            ["Air Conditioner (AC)", "Electric Heater", "Water Pump (Motor)", "Iron", "Microwave", "Washing Machine", "Electric Geyser", "EV Charger"],
            default=["Iron", "Water Pump (Motor)"]
        )

    st.write("") # Spacer

    # --- STEP 2: START ANALYSIS BUTTON ---
    if st.button("🚀 Analyze & Predict Future Usage"):
        
        # SAVE CONTEXT TO SESSION STATE
        st.session_state['household_profile'] = {
            'residents': num_people,
            'devices': heavy_devices
        }

        # A. CLEANING
        with st.spinner("⚙️ Cleaning data..."):
            df_raw = pd.read_csv(raw_file_path)
            df_clean = clean_data(df_raw)
            clean_file_path = os.path.join("data/processed", "clean_data.csv")
            df_clean.to_csv(clean_file_path, index=False)
            st.session_state['df_clean'] = df_clean
            time.sleep(0.3)
        st.success("✅ Data Cleaned")

        # B. TRAINING
        with st.spinner("🧠 Training AI Model..."):
            # UNPACK METRICS HERE
            model, feature_list, metrics = train_model(df_clean)
            st.session_state['model_metrics'] = metrics
            time.sleep(0.3)
        st.success("✅ AI Model Trained")

        # C. FORECASTING
        with st.spinner("🔮 Predicting Next 7 Days..."):
            full_future_df = predict_next_week(model, feature_list)
            
            # Filter columns
            final_columns = ['timestamp', 'hour', 'temperature_c', 'predicted_usage_kwh']
            available_cols = [c for c in final_columns if c in full_future_df.columns]
            future_df = full_future_df[available_cols]

            # Save clean version
            os.makedirs("data/prediction", exist_ok=True)
            future_df.to_csv("data/prediction/forecast_result.csv", index=False)
            st.session_state['future_df'] = future_df
        
        # --- SUB-INFORMATION SECTION (Success + Metrics) ---
        st.success("✅ Prediction Complete!")
        
        m_col1, m_col2, m_col3 = st.columns([1.5, 1.5, 4]) 
        
        with m_col1:
            st.markdown(f"""
            <div class="sub-metric-box" style="background-color: #d1fae5; color: #065f46; border: 1px solid #34d399;">
                🎯 Accuracy: {metrics['accuracy']:.1f}%
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="sub-metric-box" style="background-color: #fff7ed; color: #9a3412; border: 1px solid #fdba74;">
                ⚠️ Error: {metrics['mape']:.1f}%
            </div>
            """, unsafe_allow_html=True)
            
        st.write("") 
        # ---------------------------------------------------

        # D. VISUALIZATION GENERATION
        with st.spinner("🎨 Generating Visualizations..."):
            viz.plot_clean_daily_profile(df_clean)
            viz.plot_clean_peak_distribution(df_clean)
            viz.plot_clean_full_pattern(df_clean)
            viz.plot_clean_temp_correlation(df_clean)
            viz.plot_clean_heatmap(df_clean)

            viz.plot_pred_daily_profile(future_df)
            viz.plot_pred_peak_distribution(future_df)
            viz.plot_pred_full_forecast(future_df)
            viz.plot_pred_temp_forecast(future_df)
            viz.plot_pred_heatmap(future_df)
            
            st.session_state['analysis_done'] = True
        st.success("✅ Graphs Generated!")


# ---------------------------------------------
# 4. SHOW RESULTS DASHBOARD (Persistent)
# ---------------------------------------------
if st.session_state['analysis_done']:
    future_df = st.session_state['future_df']
    
    st.markdown("---")
    st.subheader("📊 Your AI Energy Report")
    
    # Calculate Totals
    total_kwh = future_df['predicted_usage_kwh'].sum()
    est_bill = total_kwh * 40  # Assuming 40 PKR rate
    
    # Display Big Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Predicted Load (7 Days)", value=f"{total_kwh:.2f} kWh")
    with col2:
        st.metric(label="Estimated Bill", value=f"Rs. {est_bill:,.0f}")
    with col3:
        st.metric(label="Avg Daily Usage", value=f"{total_kwh/7:.2f} kWh")

    # Display Simple Graph
    st.subheader("📈 Forecast Trend (Next 168 Hours)")
    chart_data = future_df[['timestamp', 'predicted_usage_kwh']].set_index('timestamp')
    st.line_chart(chart_data, color="#00f5d4")

    # ---------------------------------------------
    # 5. DETAILED VISUAL ANALYSIS
    # ---------------------------------------------
    st.markdown("---")
    st.subheader("🔍 Detailed Visual Analysis")
    
    graph_options = {
        "--- Historical Data (Past) ---": None,
        "Daily Load Profile (Average)": "1_clean_daily_profile.png",
        "Peak vs Off-Peak Usage": "2_clean_peak_distribution.png",
        "Full Month Usage Pattern": "3_clean_full_pattern.png",
        "Temperature vs Usage Correlation": "4_clean_temp_correlation.png",
        "Usage Intensity Heatmap": "5_clean_heatmap.png",
        
        "--- AI Forecast (Future) ---": None,
        "Predicted Daily Profile": "6_pred_daily_profile.png",
        "Predicted Peak Distribution": "7_pred_peak_distribution.png",
        "7-Day Full Forecast": "8_pred_full_forecast.png",
        "Forecast: Temp vs Usage": "9_pred_temp_forecast.png",
        "Predicted Heatmap": "10_pred_heatmap.png"
    }
    
    selected_graph_name = st.selectbox(
        "Select a Visualization to Analyze:", 
        options=[k for k in graph_options.keys()]
    )

    graph_filename = graph_options.get(selected_graph_name)
    
    if graph_filename:
        file_path = f"graphs/{graph_filename}"
        if os.path.exists(file_path):
            st.image(file_path, caption=selected_graph_name, use_container_width=True)
        else:
            st.warning("Graph file not found. Please re-run the analysis.")
    else:
        st.info("Please select a specific chart from the menu above.")

    with st.expander("📄 View Detailed Hourly Forecast Data"):
        st.dataframe(future_df)

    # ---------------------------------------------
    # 6. AI STRATEGIC PLANNER (Integrates Household Context)
    # ---------------------------------------------
st.markdown("---")
st.subheader("🤖 AI Strategic Energy Consultant")

# Automatically get the key from secrets
try:
    hf_api_key = st.secrets["HF_API_KEY"]
    st.success("✅ API Key loaded automatically from secrets.")
except Exception:
    hf_api_key = None
    st.warning("⚠️ API Key not found in secrets. Please add it to .streamlit/secrets.toml")

if hf_api_key and len(hf_api_key) > 20:
    if st.button("💡 Generate Smart Plan"):
        with st.spinner("🤖 Connecting to AI models... This may take 10-20 seconds..."):
            
            # Retrieve data
            df_clean = st.session_state.get('df_clean')
            profile = st.session_state.get('household_profile', {})
            
            if df_clean is None:
                st.error("❌ Please run the analysis first before generating AI plan.")
            else:
                try:
                    # Call AI function
                    plan = get_ai_energy_plan(
                        df_clean, 
                        future_df, 
                        hf_api_key, 
                        household_profile=profile
                    )
                    
                    # Check if error message
                    if plan.startswith("❌"):
                        st.warning(plan)
                    else:
                        st.success("✅ Strategic Plan Generated")
                        
                        # Display with nice formatting
                        st.markdown(f"""
                        <div style="background-color:#1b263b; color:#e0e1dd; padding:25px; border-radius:10px; border-left: 5px solid #00f5d4; font-family: sans-serif; line-height: 1.8; white-space: pre-wrap;">
{plan}
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Error generating plan: {str(e)}")
                    st.code(f"Error details: {type(e).__name__}")

elif hf_api_key and len(hf_api_key) <= 20:
    st.warning("⚠️ API key seems too short. Please enter a valid Hugging Face token.")
else:
    st.info("👆 Click above to expand and enter your API key to generate AI recommendations")

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("⚡ Smart AI Meter | Final Year Project")