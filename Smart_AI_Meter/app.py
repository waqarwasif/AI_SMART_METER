import streamlit as st
import pandas as pd
import numpy as np
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
from src.budget import calculate_budget_plan, calculate_cost_from_units

try:
    from analysis import visualization as viz
except ImportError:
    st.error(
        "⚠️ Could not import 'visualization'. Ensure 'visualization.py' is in the 'analysis' folder."
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

st.markdown(
    """
<div style="background-color:#0b132b; padding:20px; border-radius:10px; margin-bottom: 20px;">
    <h1 style="color:#00f5d4; margin:0;">⚡ AI Smart Energy Advisor</h1>
    <p style="color:#e0e1dd; margin:0;">Turning Energy Numbers into Human Action</p>
</div>
""",
    unsafe_allow_html=True,
)

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
# 3. Global Status Widget
# ---------------------------------------------
if "savings_streak" not in st.session_state:
    st.session_state.savings_streak = 5

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
# 4. Live IoT Monitor
# ---------------------------------------------
st.subheader("📡 Live IoT Monitor")
st.caption(
    "Real-time stream from Smart Meter (DEMO MODE: Simulating Hardware Connection)"
)

if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame(
        columns=["timestamp", "voltage", "current", "power_kw"]
    )

if st.toggle("🔌 Activate IoT Simulation Mode"):
    placeholder = st.empty()
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
            k1, k2, k3 = st.columns(3)
            k1.metric(
                "Live Load",
                f"{df_display['power_kw'].iloc[-1]:.2f} kW",
                delta_color="inverse",
            )
            k2.metric("Voltage", f"{df_display['voltage'].iloc[-1]:.1f} V")
            k3.metric("Grid Status", "ONLINE ⚡")
            st.area_chart(df_display["power_kw"], color="#00f5d4", height=200)
        time.sleep(0.5)

st.markdown("---")

# ---------------------------------------------
# 5. MAIN APPLICATION TABS
# ---------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Prediction & Strategy", "💰 Reverse Budget", "☀️ Solar ROI", "💬 AI Assistant"]
)

# =========================================
# TAB 1: DIAGNOSIS -> PRESCRIPTION WORKFLOW
# =========================================
with tab1:
    uploaded_file = st.file_uploader(
        "📂 Upload electricity usage CSV", type=["csv"], key="upload_ui"
    )

    # Initialize Session State
    for key in [
        "analysis_done",
        "household_profile",
        "df_clean",
        "ai_plan",
        "agent_plan",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    if uploaded_file:
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("graphs", exist_ok=True)

        # Save file
        raw_file_path = os.path.join("data/raw", uploaded_file.name)
        with open(raw_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

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
                    "Refrigerator",
                    "Gaming PC",
                ],
                default=["Iron", "Motor"],
            )

        # --- STEP 1: THE DIAGNOSIS (Prediction) ---
        if st.button("🚀 Analyze Current Status"):
            st.session_state["household_profile"] = {
                "residents": num_people,
                "devices": heavy_devices,
                "season": season_input,
            }

            with st.spinner("⚙️ Running AI Diagnostics..."):
                # Cleaning
                df_raw = pd.read_csv(raw_file_path)
                df_clean = clean_data(df_raw)
                st.session_state["df_clean"] = df_clean

                # Training
                model, feature_list, metrics = train_model(df_clean)
                st.session_state["model_metrics"] = metrics

                # Forecasting
                full_future_df = predict_next_week(model, feature_list)
                st.session_state["future_df"] = full_future_df

                # Visualization Generation
                viz.plot_clean_daily_profile(df_clean)
                viz.plot_pred_daily_profile(full_future_df)
                # (Add other viz calls here as needed)

                st.session_state["analysis_done"] = True
                st.rerun()  # Refresh to show results

    # --- RESULTS & OPTIMIZATION SECTION ---
    if st.session_state.get("analysis_done"):
        future_df = st.session_state["future_df"]
        total_kwh = future_df["predicted_usage_kwh"].sum()
        est_bill = calculate_cost_from_units(total_kwh)

        # 1. THE DIAGNOSIS (Metrics)
        st.markdown("---")
        st.subheader("📊 The Diagnosis")

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Load (7 Days)", f"{total_kwh:.2f} kWh")
        m2.metric("Projected Bill (Approx)", f"Rs. {est_bill:,.0f}")
        m3.metric("Avg Daily Usage", f"{total_kwh/7:.2f} kWh")

        st.line_chart(
            future_df[["timestamp", "predicted_usage_kwh"]].set_index("timestamp")
        )

        # 2. THE PRESCRIPTION (Optimization Studio)
        st.markdown("---")
        st.subheader("🤖 AI Optimization Studio")
        st.info(
            f"Based on your prediction of **{total_kwh:.0f} Units**, do you want to optimize this?"
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            # Pre-fill with estimated bill so user sees context
            target_budget = st.number_input(
                "💰 Set Your Target Budget (Rs.)",
                value=int(est_bill),
                step=500,
                help="Lower this number to see how to save money.",
            )
        with col_opt2:
            st.caption(
                "The AI will calculate exactly which devices to cut to hit your target."
            )

        # --- THE AGENT TRIGGER ---
        if st.button("✨ Generate AI Savings Plan"):

            # A. Run the Agent (Budget Logic)
            user_devices = st.session_state["household_profile"].get("devices", [])
            agent_plan = calculate_budget_plan(
                target_bill_rs=target_budget,
                predicted_kwh=total_kwh,
                user_selected_devices=user_devices,
            )
            st.session_state["agent_plan"] = agent_plan  # Save for Chatbot

            # B. Display Agent Results
            if agent_plan["status"] == "SAFE":
                st.success(
                    "✅ **You are safe!** Your target is higher than your predicted usage."
                )
            else:
                st.warning(
                    f"⚠️ **Action Needed!** You need to save **{agent_plan['gap_units']} kWh** to hit your budget."
                )
                with st.expander("📋 View Detailed Action Plan", expanded=True):
                    for action in agent_plan["actions"]:
                        st.write(f"- {action}")

            # C. Generate Report (Recommender)
            try:
                hf_api_key = st.secrets.get("HF_API_KEY", "")
            except:
                hf_api_key = ""

            if not hf_api_key:
                hf_api_key = st.text_input(
                    "🔑 Enter Hugging Face API Key (Required for PDF Report)",
                    type="password",
                    key="pdf_key_input",
                )

            if hf_api_key:
                with st.spinner("Writing Official Report..."):
                    plan_text = get_ai_energy_plan(
                        st.session_state["df_clean"],
                        st.session_state["future_df"],
                        hf_api_key,
                        st.session_state["household_profile"],
                        agent_plan=agent_plan,
                    )
                    st.session_state["ai_plan"] = plan_text

        # Report Download
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
# TAB 2: REVERSE BUDGET PLANNER (Fixed)
# =========================================
with tab2:
    st.subheader("📉 Reverse Budget Calculator")
    st.info("Input your money budget to see your daily energy limit.")

    c1, c2 = st.columns(2)
    with c1:
        target_budget_legacy = st.number_input(
            "💰 Monthly Budget (Rs.)", value=5000, step=500
        )
    with c2:
        days_left = st.slider("🗓️ Days Remaining", 1, 30, 20)

    # Optional: Input current usage if known (defaults to 0 for a clean start)
    current_used = st.number_input(
        "⚡ Units Used So Far (Optional)",
        value=0,
        step=10,
        help="Check your meter reading if possible.",
    )

    if st.button("Calculate Daily Limit"):

        # Get devices from profile (or default list)
        user_devices = st.session_state.get("household_profile", {}).get(
            "devices", ["AC", "Iron", "Fans"]
        )

        # Call Budget Agent in "Calculator Mode" (predicted_kwh=0)
        plan = calculate_budget_plan(
            target_bill_rs=target_budget_legacy,
            current_usage_kwh=current_used,
            days_left=days_left,
            user_selected_devices=user_devices,
            predicted_kwh=0,  # Explicitly 0 triggers Calculator Mode
        )

        # Display Results
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Allowed Total Units", f"{plan['target_units']} kWh")
        col_res2.metric("Safe Daily Limit", f"{plan['daily_limit']} kWh/day")

        if plan["status"] == "CRITICAL":
            st.error(plan["message"])
        else:
            st.success(plan["message"])

        st.caption("Based on your budget, here is what you can run daily:")
        for action in plan.get("actions", []):
            st.write(action)

# =========================================
# TAB 3: SOLAR ROI
# =========================================
with tab3:
    st.subheader("☀️ Solar Investment Planner")

    # 1. Validation: Added min_value=0 to prevent negative inputs
    bill_input = st.number_input(
        "Average Monthly Bill (Rs.)", value=25000, step=1000, min_value=0
    )

    # 2. Logic Check: Only show button if bill is valid
    if bill_input > 0:
        if st.button("Calculate ROI"):
            solar_data = calculate_solar_roi(bill_input)

            c1, c2, c3 = st.columns(3)
            c1.metric("System Size", f"{solar_data['system_size_kw']} kW")
            c2.metric("Total Cost", f"Rs. {solar_data['total_cost']:,.0f}")
            c3.metric("Payback", f"{solar_data['payback_years']:.1f} Years")

            st.bar_chart(solar_data["chart_data"].set_index("Year"))
            st.success(
                f"✅ Free electricity after {solar_data['payback_years']:.1f} years!"
            )
    else:
        st.info(
            "👋 Enter your monthly bill amount above to see your solar savings potential."
        )

# =========================================
# TAB 4: CONTEXT-AWARE AI CHATBOT
# =========================================
with tab4:
    st.subheader("💬 Energy AI Assistant")
    st.caption(
        "Ask specific questions about your plan. The AI knows your prediction and budget."
    )

    try:
        hf_api_key = st.secrets["HF_API_KEY"]
    except:
        hf_api_key = st.text_input(
            "Enter Hugging Face API Key for Chat", type="password"
        )

    if hf_api_key:
        # 1. Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 2. Display History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 3. Handle Input
        if user_input := st.chat_input("Type your question here..."):
            st.chat_message("user").markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            # 4. Prepare Context (The Brain)
            agent_context = ""
            if "agent_plan" in st.session_state and st.session_state["agent_plan"]:
                plan = st.session_state["agent_plan"]
                agent_context = f"""
                [LIVE SYSTEM DATA]
                - Target Budget: {plan.get('target_units', 0)} kWh
                - Projected Usage: {plan.get('predicted_units', 0)} kWh
                - Current Status: {plan.get('status', 'Unknown')}
                - REQUIRED ACTIONS: {', '.join(plan.get('actions', []))}
                """

            # 5. Build Memory Chain (Sliding Window)
            history_for_ai = [
                {
                    "role": "system",
                    "content": f"You are a helpful energy expert. Use this live system data to answer: {agent_context}. Answer short and concise.",
                }
            ]
            # Take last 10 messages for memory efficiency
            history_for_ai.extend(st.session_state.messages[-10:])

            with st.spinner("Thinking..."):
                try:
                    client = InferenceClient(token=hf_api_key)
                    response = client.chat_completion(
                        model="meta-llama/Llama-3.2-3B-Instruct",
                        messages=history_for_ai,
                        max_tokens=500,
                        temperature=0.3,
                    )
                    reply = response.choices[0].message.content
                    st.chat_message("assistant").markdown(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
                except Exception as e:
                    st.error(f"AI Error: {e}")

st.markdown("---")
st.caption("⚡ Smart AI Meter | Energy Usage Advisor Project")
