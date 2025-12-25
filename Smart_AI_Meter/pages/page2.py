# page2.py
import streamlit as st

st.set_page_config(page_title="7-Day Plan", layout="wide")

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;'>7-Day Plan Overview</h1>",
    unsafe_allow_html=True
)

# ---------------- CSS ----------------
st.markdown("""
<style>

/* Neon card styling */
.neon-card {
    background: #0b132b;
    border: 2px solid #4cc9f0;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    color: #00f5d4;
    box-shadow: 0 0 8px #4cc9f0, 0 0 16px #00f5d4;
    animation: pulse 2.5s infinite alternate;
    cursor: pointer;
}

/* Hover effect */
.neon-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px #4cc9f0, 0 0 40px #00f5d4;
}

/* Pulse animation */
@keyframes pulse {
    from { box-shadow: 0 0 6px #4cc9f0, 0 0 12px #00f5d4; }
    to { box-shadow: 0 0 14px #4cc9f0, 0 0 28px #00f5d4; }
}

</style>
""", unsafe_allow_html=True)

# ---------------- STATE ----------------
if "selected_day" not in st.session_state:
    st.session_state.selected_day = "Day 1"

# ---------------- LAYOUT ----------------
left_col, _ = st.columns([1, 3])  # Only left column, right side removed

# ========== LEFT SIDE (NEON CARDS) ==========
with left_col:
    st.subheader("7-Day Plan")
    days = [f"Day {i}" for i in range(1, 8)]

    for day in days:
        # Highlight selected day
        if st.session_state.selected_day == day:
            st.markdown(
                f"<div class='neon-card' style='border-color:#00f5d4;'>{day}<br><small>Plan Placeholder</small></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='neon-card'>{day}<br><small>Plan Placeholder</small></div>",
                unsafe_allow_html=True
            )