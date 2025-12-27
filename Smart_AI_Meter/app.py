import streamlit as st
import pandas as pd
import os
import time
from fpdf import FPDF

# --- IMPORT YOUR MODULES ---
from src.processor import clean_data
from src.predictor import train_model
from src.forecaster import predict_next_week
from src.recommender import get_ai_energy_plan

# Import the visualization module
try:
    from analysis import visualization as viz
except ImportError:
    st.error("⚠️ Could not import 'visualization'. Please ensure 'visualization.py' is in the 'analysis' folder.")

# ---------------------------------------------
# 0. PDF Generator Function (FIXED FOR UNICODE)
# ---------------------------------------------
def create_pdf(report_text):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'AI Smart Energy Report', 0, 1, 'C')
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    # 1. Sanitize Text (Remove Emojis & Fix Bullets)
    # FPDF only supports Latin-1, so we replace fancy chars with simple ones
    replacements = {
        '\u2022': '-',      # Bullet point -> Dash
        '\u2013': '-',      # En dash -> Dash
        '\u2014': '-',      # Em dash -> Dash
        '\u2018': "'",      # Smart quote -> '
        '\u2019': "'",      # Smart quote -> '
        '\u201c': '"',      # Smart quote -> "
        '\u201d': '"',      # Smart quote -> "
        '⚡': '', '🏠': '', '✅': '[OK]', '⚠️': '[WARN]', '🚨': '[ALERT]', 
        '❄️': '', '💧': '', '👕': '', '💡': '', '🤖': '', '📊': '', 
        '📉': '', '💰': '', '🗓️': '', '📑': ''
    }
    
    safe_text = report_text
    for char, replacement in replacements.items():
        safe_text = safe_text.replace(char, replacement)
    
    # Final cleanup: Remove any other non-latin characters
    safe_text = safe_text.encode('latin-1', 'replace').decode('latin-1')

    # 2. Generate PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add content
    # We strip markdown stars/hashes for cleaner plain text in PDF
    clean_text = safe_text.replace('**', '').replace('###', '').replace('####', '')
    
    pdf.multi_cell(0, 7, clean_text)
    
    # Return binary
    return pdf.output(dest='S').encode('latin-1')

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

/* --- NEW: PAPER REPORT STYLING --- */
.paper-sheet {
    background-color: #ffffff;       /* White Paper Background */
    color: #2c3e50;                  /* Dark Grey Ink Color for readability */
    padding: 40px 50px;              /* Generous "Print" Margins */
    border-radius: 2px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15); /* Drop Shadow to make it "float" */
    font-family: 'Georgia', 'Times New Roman', serif; /* Professional Serif Font */
    font-size: 16px;
    line-height: 1.6;
    
    /* Scrollable Settings */
    max-height: 600px;               /* Fixed Height */
    overflow-y: auto;                /* Internal Scrollbar */
    border: 1px solid #dcdcdc;
}

/* Make the scrollbar look elegant inside the paper */
.paper-sheet::-webkit-scrollbar {
    width: 10px;
}
.paper-sheet::-webkit-scrollbar-track {
    background: #f1f1f1; 
}
.paper-sheet::-webkit-scrollbar-thumb {
    background: #888; 
    border-radius: 5px;
}
.paper-sheet::-webkit-scrollbar-thumb:hover {
    background: #555; 
}

/* Header style inside the paper */
.paper-header {
    text-align: center;
    border-bottom: 2px solid #2c3e50;
    margin-bottom: 20px;
    padding-bottom: 10px;
    color: #0b132b;
    font-family: 'Arial', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1px;
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
if 'ai_plan' not in st.session_state:
    st.session_state['ai_plan'] = None

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
        
        # --- NEW: SEASON DROPDOWN ---
        season_input = st.selectbox(
            "🌤️ Select Season Context:",
            [
                "Winter (Heating/Geyser Season)", 
                "Summer (Peak AC Season)", 
                "Spring (Moderate)", 
                "Autumn (Transition)"
            ],
            index=1 # Default to Summer
        )
    
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
        
        # SAVE CONTEXT TO SESSION STATE (Including User Selected Season)
        st.session_state['household_profile'] = {
            'residents': num_people,
            'devices': heavy_devices,
            'season': season_input  # <--- Saving Season
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
            st.image(
                        file_path,
                        caption=selected_graph_name,
                        width=900   # 👈 TRUE VISUAL SIZE CONTROL
                    )

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

# Automatically get the key from secrets or let user enter it
try:
    hf_api_key = st.secrets["HF_API_KEY"]
except Exception:
    hf_api_key = ""

# If no secret, show input box
if not hf_api_key:
    hf_api_key = st.text_input("Enter Hugging Face API Key", type="password")

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
                        st.session_state['future_df'], 
                        hf_api_key, 
                        household_profile=profile
                    )
                    
                    st.session_state['ai_plan'] = plan
                    
                    # Check if error message
                    if plan.startswith("❌"):
                        st.warning(plan)
                    else:
                        st.success("✅ Strategic Plan Generated")
                
                except Exception as e:
                    st.error(f"❌ Error generating plan: {str(e)}")

    # DISPLAY THE REPORT IF IT EXISTS
    if st.session_state['ai_plan'] and not st.session_state['ai_plan'].startswith("❌"):
        plan = st.session_state['ai_plan']
        
        st.markdown("---")
        
        # --- NEW: CLICKABLE DOCUMENT UI (Paper Look) ---
        with st.expander("📄 𝐂𝐋𝐈𝐂𝐊 𝐓𝐎 𝐎𝐏𝐄𝐍: Weekly Energy Strategy Report", expanded=False):
            
            # The "Paper" Container
            st.markdown(f"""
            <div class="paper-sheet">
                <h2 class="paper-header">
                    ⚡ Energy Audit & Strategy Report
                </h2>
                {plan}
                <br><br>
                <div style="text-align: center; font-size: 12px; color: #aaa; border-top: 1px solid #eee; padding-top: 10px;">
                    Generated by AI Smart Meter Advisor • CONFIDENTIAL
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Download Button (Inside the expander, below the paper)
            col_dl_1, col_dl_2, col_dl_3 = st.columns([1, 2, 1])
            with col_dl_2:
                st.write("") # Spacer
                pdf_bytes = create_pdf(plan)
                st.download_button(
                    label="📥 Download This Report as PDF",
                    data=pdf_bytes,
                    file_name="AI_Energy_Report.pdf",
                    mime="application/pdf",
                    key='download-pdf',
                    help="Save this strategy to your device"
                )

elif hf_api_key and len(hf_api_key) <= 20:
    st.warning("⚠️ API key seems too short. Please enter a valid Hugging Face token.")
else:
    st.info("👆 Enter your API key above to generate AI recommendations")

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("⚡ Smart AI Meter | Energy Usage Advisor Project")