import streamlit as st
import pandas as pd
import numpy as np  # Needed for simulation
import os
import time
from fpdf import FPDF
from huggingface_hub import InferenceClient

# --- IMPORTING MODULES ---
from src.processor import clean_data
from src.predictor import train_model
from src.forecaster import predict_next_week
from src.recommender import get_ai_energy_plan
from src.solar import calculate_solar_roi
from src.budget import calculate_budget_plan

# Import the visualization module
try:
    from analysis import visualization as viz
except ImportError:
    st.error(
        "⚠️ Could not import 'visualization'. Please ensure 'visualization.py' is in the 'analysis' folder."
    )


# ---------------------------------------------
# 0. PDF Generator Function
# ---------------------------------------------
def create_pdf(report_text):
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 15)
            self.cell(0, 10, "AI Smart Energy Report", 0, 1, "C")
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    # Sanitize Text
    replacements = {
        "\u2022": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "⚡": "",
        "🏠": "",
        "✅": "[OK]",
        "⚠️": "[WARN]",
        "🚨": "[ALERT]",
        "❄️": "",
        "💧": "",
        "👕": "",
        "💡": "",
        "🤖": "",
        "📊": "",
        "📉": "",
        "💰": "",
        "🗓️": "",
        "📑": "",
    }

    safe_text = report_text
    for char, replacement in replacements.items():
        safe_text = safe_text.replace(char, replacement)

    safe_text = safe_text.encode("latin-1", "replace").decode("latin-1")

    # Generate PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    clean_text = safe_text.replace("**", "").replace("###", "").replace("####", "")
    pdf.multi_cell(0, 7, clean_text)
    return pdf.output(dest="S").encode("latin-1")


# ---------------------------------------------
# 1. Page Configuration
# ---------------------------------------------
st.set_page_config(page_title="⚡ AI Smart Energy Advisor", layout="wide")

# ---------------------------------------------
# 2. UI: Hero Section & Styling
# ---------------------------------------------
st.markdown(
    """
<div style="background-color:#0b132b; padding:20px; border-radius:10px; margin-bottom: 20px;">
    <h1 style="color:#00f5d4; margin:0;">⚡ AI Smart Energy Advisor</h1>
    <p style="color:#e0e1dd; margin:0;">Turning Energy Numbers into Human Action</p>
</div>
""",
    unsafe_allow_html=True,
)

# --- CUSTOM CSS ---
st.markdown(
    """
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
.paper-sheet {
    background-color: #ffffff;
    color: #2c3e50;
    padding: 40px 50px;
    border-radius: 2px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    font-family: 'Georgia', 'Times New Roman', serif;
    font-size: 16px;
    line-height: 1.6;
    max-height: 600px;
    overflow-y: auto;
    border: 1px solid #dcdcdc;
}
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
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------
# 3. GLOBAL STATUS WIDGET (Traffic Light & Streak)
# ---------------------------------------------
if "savings_streak" not in st.session_state:
    st.session_state.savings_streak = 5  # Demo value

# Simple Logic for Traffic Light (Can be connected to real daily usage later)
usage_status = "SAFE"

col_status1, col_status2 = st.columns([3, 1])
with col_status1:
    if usage_status == "SAFE":
        st.success("🟢 DAILY STATUS: GREEN (You are under your daily budget)")
    else:
        st.error("🔴 DAILY STATUS: RED (High usage detected!)")

with col_status2:
    st.metric("🔥 Savings Streak", f"{st.session_state.savings_streak} Days")

st.markdown("---")

# ---------------------------------------------
# 4. LIVE IOT MONITOR (Cloud Compatible Version)
# ---------------------------------------------
st.subheader("📡 Live IoT Monitor")
st.caption(
    "Real-time stream from Smart Meter (DEMO MODE: Simulating Hardware Connection)"
)

# Initialize session state for live data if not exists
if "live_data" not in st.session_state:
    # Create empty dataframe with structure
    st.session_state.live_data = pd.DataFrame(
        columns=["timestamp", "voltage", "current", "power_kw"]
    )

if st.toggle("🔌 Activate IoT Simulation Mode"):
    placeholder = st.empty()

    # Run the loop for 50 iterations (approx 25 seconds) to prevent cloud timeout
    # User can toggle off/on to restart
    for _ in range(50):

        
        now = pd.Timestamp.now()
        voltage = np.random.normal(220, 2)
        current = np.random.normal(8, 3)
        
        if np.random.random() > 0.8:
            current += 10
        power = (voltage * current) / 1000

        new_row = pd.DataFrame(
            [[now, voltage, current, power]],
            columns=["timestamp", "voltage", "current", "power_kw"],
        )

        st.session_state.live_data = pd.concat(
            [st.session_state.live_data, new_row], ignore_index=True
        )

        if len(st.session_state.live_data) > 50:
            st.session_state.live_data = st.session_state.live_data.tail(50)

        with placeholder.container():
            df_display = st.session_state.live_data

            last_power = df_display["power_kw"].iloc[-1]
            last_volts = df_display["voltage"].iloc[-1]

            k1, k2, k3 = st.columns(3)
            k1.metric("Live Load", f"{last_power:.2f} kW", delta_color="inverse")
            k2.metric("Voltage", f"{last_volts:.1f} V")
            k3.metric("Grid Status", "ONLINE ⚡")

            st.area_chart(df_display["power_kw"], color="#00f5d4", height=200)
            st.caption("Displaying real-time power fluctuation (Generated In-App).")

        time.sleep(0.5)  

st.markdown("---")

# ---------------------------------------------
# 5. MAIN APPLICATION TABS
# ---------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Prediction Dashboard", "💰 Budget Planner", "☀️ Solar ROI", "💬 AI Chatbot"]
)

# =========================================
# TAB 1: PREDICTION DASHBOARD (Original App)
# =========================================
with tab1:
    uploaded_file = st.file_uploader(
        "📂 Upload electricity usage CSV", type=["csv"], key="upload_ui"
    )

    # Initialize session state
    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False
    if "household_profile" not in st.session_state:
        st.session_state["household_profile"] = {}
    if "df_clean" not in st.session_state:
        st.session_state["df_clean"] = None
    if "ai_plan" not in st.session_state:
        st.session_state["ai_plan"] = None

    if uploaded_file:
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("graphs", exist_ok=True)

        raw_file_path = os.path.join("data/raw", uploaded_file.name)
        with open(raw_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ File received: {uploaded_file.name}")

        st.markdown("### 🏠 Household Context")
        c1, c2 = st.columns(2)
        with c1:
            num_people = st.number_input("👨‍👩‍👧‍👦 Residents", 1, 20, 4)
            season_input = st.selectbox(
                "🌤️ Season",
                ["Winter (Heating)", "Summer (AC)", "Spring", "Autumn"],
                index=1,
            )
        with c2:
            heavy_devices = st.multiselect(
                "🔌 High-Load Devices",
                [
                    "AC",
                    "Heater",
                    "Motor",
                    "Iron",
                    "Microwave",
                    "Washing Machine",
                    "Geyser",
                    "EV Charger",
                ],
                default=["Iron", "Motor"],
            )

        if st.button("🚀 Analyze & Predict"):
            st.session_state["household_profile"] = {
                "residents": num_people,
                "devices": heavy_devices,
                "season": season_input,
            }

            # A. CLEANING
            with st.spinner("⚙️ Cleaning data..."):
                df_raw = pd.read_csv(raw_file_path)
                df_clean = clean_data(df_raw)
                clean_file_path = os.path.join("data/processed", "clean_data.csv")
                df_clean.to_csv(clean_file_path, index=False)
                st.session_state["df_clean"] = df_clean

            # B. TRAINING
            with st.spinner("🧠 Training AI Model..."):
                model, feature_list, metrics = train_model(df_clean)
                st.session_state["model_metrics"] = metrics

            # C. FORECASTING
            with st.spinner("🔮 Predicting Next 7 Days..."):
                full_future_df = predict_next_week(model, feature_list)

                # Filter columns
                final_columns = [
                    "timestamp",
                    "hour",
                    "temperature_c",
                    "predicted_usage_kwh",
                ]
                available_cols = [
                    c for c in final_columns if c in full_future_df.columns
                ]
                future_df = full_future_df[available_cols]

                # Save clean version
                os.makedirs("data/prediction", exist_ok=True)
                future_df.to_csv("data/prediction/forecast_result.csv", index=False)
                st.session_state["future_df"] = future_df

            # --- SUB-INFORMATION SECTION (Success + Metrics) ---
            st.success("✅ Prediction Complete!")

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(
                    f'<div class="sub-metric-box" style="background-color: #d1fae5; color: #065f46; border: 1px solid #34d399;">🎯 Accuracy: {metrics["accuracy"]:.1f}%</div>',
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f'<div class="sub-metric-box" style="background-color: #fff7ed; color: #9a3412; border: 1px solid #fdba74;">⚠️ Error: {metrics["mape"]:.1f}%</div>',
                    unsafe_allow_html=True,
                )

            # D. VISUALIZATION
            with st.spinner("🎨 Generating Graphs..."):
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

                st.session_state["analysis_done"] = True

            st.success("✅ Analysis Complete!")

    # SHOW RESULTS
    if st.session_state["analysis_done"]:
        future_df = st.session_state["future_df"]
        total_kwh = future_df["predicted_usage_kwh"].sum()
        est_bill = total_kwh * 40

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Load (7 Days)", f"{total_kwh:.2f} kWh")
        c2.metric("Estimated Bill", f"Rs. {est_bill:,.0f}")
        c3.metric("Avg Daily Usage", f"{total_kwh/7:.2f} kWh")

        st.line_chart(
            future_df[["timestamp", "predicted_usage_kwh"]].set_index("timestamp")
        )

        # Visuals Section
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
            "Predicted Heatmap": "10_pred_heatmap.png",
        }

        selected_graph_name = st.selectbox(
            "Select a Visualization to Analyze:",
            options=[k for k in graph_options.keys()],
        )
        graph_filename = graph_options.get(selected_graph_name)

        if graph_filename:
            file_path = f"graphs/{graph_filename}"
            if os.path.exists(file_path):
                st.image(file_path, caption=selected_graph_name, width=900)

        # AI REPORT GENERATION
        st.markdown("### 🤖 AI Strategy Report")
        try:
            hf_api_key = st.secrets["HF_API_KEY"]
        except:
            hf_api_key = st.text_input("Enter Hugging Face API Key", type="password")

        if hf_api_key and st.button("💡 Generate Smart Plan"):
            with st.spinner("Writing report..."):
                plan = get_ai_energy_plan(
                    st.session_state["df_clean"],
                    future_df,
                    hf_api_key,
                    st.session_state["household_profile"],
                )
                st.session_state["ai_plan"] = plan

        if st.session_state.get("ai_plan"):
            with st.expander("📄 Click to Open Report"):
                st.markdown(
                    f'<div class="paper-sheet"><h2 class="paper-header">Energy Strategy</h2>{st.session_state["ai_plan"]}</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    "📥 Download PDF",
                    create_pdf(st.session_state["ai_plan"]),
                    "Report.pdf",
                )

# =========================================
# TAB 2: REVERSE BUDGET PLANNER
# =========================================
with tab2:
    st.subheader("📉 Reverse Budget Calculator")
    st.info("Tell us your budget, we tell you how to survive.")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        target_budget = st.number_input(
            "My Budget for this Month (Rs.)", value=10000, step=500
        )
    with b_col2:
        days_left = st.slider("Days remaining in month", 1, 30, 15)

    if st.button("Generate Survival Plan"):
        # Dummy current usage for demo purposes (In real app, calculate from CSV)
        plan = calculate_budget_plan(
            target_budget, current_usage_kwh=150, days_left=days_left
        )

        st.metric("Safe Daily Limit", f"{plan['daily_limit']:.1f} kWh")
        if plan["status"] == "SAFE":
            st.success(plan["message"])
        else:
            st.error(plan["message"])

# =========================================
# TAB 3: SOLAR ROI CALCULATOR
# =========================================
with tab3:
    st.subheader("☀️ Solar Investment Planner")
    st.caption("Calculate your potential savings and payback period.")

    bill_input = st.number_input("Average Monthly Bill (Rs.)", value=25000, step=1000)

    if st.button("Calculate ROI"):
        solar_data = calculate_solar_roi(bill_input)

        c1, c2, c3 = st.columns(3)
        c1.metric("Recommended System", f"{solar_data['system_size_kw']} kW")
        c2.metric("Total Cost", f"Rs. {solar_data['total_cost']:,.0f}")
        c3.metric("Payback Period", f"{solar_data['payback_years']:.1f} Years")

        st.bar_chart(solar_data["chart_data"].set_index("Year"))
        st.success(
            f"✅ After {solar_data['payback_years']:.1f} years, your electricity is effectively FREE!"
        )

# =========================================
# TAB 4: AI CHATBOT
# =========================================
with tab4:
    st.subheader("💬 Energy AI Assistant")
    st.caption("Ask questions like: 'Why is my bill high?' or 'How to save money?'")

    # Get API Key again if needed
    try:
        hf_api_key = st.secrets["HF_API_KEY"]
    except:
        hf_api_key = ""

    if not hf_api_key:
        st.warning("Please enter your API Key in the Dashboard tab first.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_input := st.chat_input("Type your question here..."):
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.spinner("Thinking..."):
                try:
                    client = InferenceClient(token=hf_api_key)
                    # Simple context for the AI
                    context = f"User has {num_people if 'num_people' in locals() else 'unknown'} residents. Season is {season_input if 'season_input' in locals() else 'unknown'}."

                    response = client.chat_completion(
                        model="meta-llama/Llama-3.2-3B-Instruct",
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are a helpful energy expert. {context}. Answer short and concise.",
                            },
                            {"role": "user", "content": user_input},
                        ],
                        max_tokens=300,
                    )
                    reply = response.choices[0].message.content
                    st.chat_message("assistant").markdown(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
                except Exception as e:
                    st.error(f"AI Error: {e}")

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("⚡ Smart AI Meter | Energy Usage Advisor Project")
