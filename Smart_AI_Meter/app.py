import streamlit as st
import pandas as pd
import os
from src.processor import clean_data

# ---------------------------------------------
# 1. Page Configuration (From ui_components.py)
# ---------------------------------------------
st.set_page_config(
    page_title="⚡ AI Smart Energy Advisor",
    layout="wide"
)

# ---------------------------------------------
# 2. UI: Hero Section & Styling (From ui_components.py)
# ---------------------------------------------
st.markdown("""
<div style="background-color:#0b132b; padding:20px; border-radius:10px; margin-bottom: 20px;">
<h1 style="color:#00f5d4; margin:0;">⚡ AI Smart Energy Advisor</h1>
<p style="color:#e0e1dd; margin:0;">Turning Energy Numbers into Human Action</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Custom CSS for the button (From ui_components.py)
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
}
.stButton > button:hover {
    box-shadow: 0 0 20px #00f5d4;
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# 3. Logic: Upload -> Save -> Clean -> Save
# ---------------------------------------------
uploaded_file = st.file_uploader("📂 Upload electricity usage CSV", type=["csv"], key="upload_ui")

if uploaded_file:
    # A. Create directories if they don't exist
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # B. Save Raw File
    raw_file_path = os.path.join("data/raw", uploaded_file.name)
    with open(raw_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"✅ Raw file saved to: `{raw_file_path}`")

    # C. Process Data
    # We use a button or do it automatically. 
    # Based on your flow, doing it automatically upon upload is often smoother, 
    # but let's use a spinner to show it's happening.
    
    with st.spinner("⚙️ Cleaning data (Removing glitches, handling outliers)..."):
        # 1. Read
        df_raw = pd.read_csv(raw_file_path)
        
        # 2. Clean (using your processor.py)
        df_clean = clean_data(df_raw)
        
        # 3. Save Clean File
        clean_file_path = os.path.join("data/processed", "clean_data.csv")
        df_clean.to_csv(clean_file_path, index=False)

    st.success(f"✅ Data cleaned and saved to: `{clean_file_path}`")
    
    # Optional: Show a preview (Text only, no graphs as requested)
    with st.expander("👀 View Cleaned Data Preview"):
        st.dataframe(df_clean.head())

# ---------------------------------------------
# Footer
# ---------------------------------------------
st.markdown("---")
st.caption("⚡ Smart AI Meter | Smart Energy Project")