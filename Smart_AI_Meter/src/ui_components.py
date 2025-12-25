import streamlit as st

# ---------------------------------------------
# Page Title board
# ---------------------------------------------
st.set_page_config(
    page_title="⚡ AI Smart Energy Advisor",
    layout="wide"
)


# ---------------------------------------------
#  Hero Section
# ---------------------------------------------
st.markdown("""
<div style="background-color:#0b132b; padding:20px; border-radius:10px">
<h1 style="color:#00f5d4">⚡ AI Smart Energy Advisor</h1>
<p style="color:#e0e1dd">Turning Energy Numbers into Human Action</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------
# Upload CSV
# ---------------------------------------------
uploaded_file = st.file_uploader("📂 Upload electricity usage CSV", type=["csv"], key="upload_ui")

#Button
# ---------------------------------------------
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

if uploaded_file:
    if st.button("⚡ Generate 7-Day Energy Plan"):
        st.switch_page("pages/page2.py")

# ---------------------------------------------


# Footer
st.markdown("---")
st.caption("⚡ Smart AI Meter | Smart Energy Project")
