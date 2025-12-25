import streamlit as st
import pandas as pd
import os
import time

# --- IMPORT YOUR MODULES ---
from src.processor import clean_data
from src.predictor import train_model       # <--- Added
from src.forecaster import predict_next_week # <--- Added

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
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 3. Logic: The Full Pipeline
# ---------------------------------------------
uploaded_file = st.file_uploader("📂 Upload electricity usage CSV", type=["csv"], key="upload_ui")

if uploaded_file:
    # --- STEP 1: SAVE RAW FILE ---
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    raw_file_path = os.path.join("data/raw", uploaded_file.name)
    with open(raw_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ Raw file received")

    # --- STEP 2: START ANALYSIS BUTTON ---
    if st.button("🚀 Analyze & Predict Future Usage"):
        
        # A. CLEANING
        with st.spinner("⚙️ Cleaning data (Removing glitches, handling outliers)..."):
            df_raw = pd.read_csv(raw_file_path)
            df_clean = clean_data(df_raw)
            
            clean_file_path = os.path.join("data/processed", "clean_data.csv")
            df_clean.to_csv(clean_file_path, index=False)
            time.sleep(1) # Just for effect
        st.success("✅ Data Cleaned")

        # B. TRAINING
        with st.spinner("🧠 Training AI Model on your patterns..."):
            # This calls your predictor.py
            model, feature_list = train_model(df_clean)
            time.sleep(1)
        st.success("✅ AI Model Trained")

        # C. FORECASTING
        with st.spinner("🔮 Fetching Weather & Predicting Next 7 Days..."):
            # This calls your forecaster.py
            future_df = predict_next_week(model, feature_list)
        st.success("✅ Prediction Complete!")

        # ---------------------------------------------
        # 4. SHOW RESULTS DASHBOARD
        # ---------------------------------------------
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

        # Display Graph
        st.subheader("📈 Forecast Trend (Next 168 Hours)")
        
        # Create a clean chart data
        chart_data = future_df[['timestamp', 'predicted_usage_kwh']].set_index('timestamp')
        st.line_chart(chart_data, color="#00f5d4")

        # Display Data Table
        with st.expander("📄 View Detailed Hourly Forecast"):
            st.dataframe(future_df[['timestamp', 'hour', 'temperature_c', 'predicted_usage_kwh']])

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("⚡ Smart AI Meter | Smart Energy Project")