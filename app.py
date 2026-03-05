st.markdown("""
    <style>
    .stToolbar { display: none !important; }
    </style>
""", unsafe_allow_html=True)

import streamlit as st

# Page title
st.set_page_config(
    page_title="Prediction Tools",
    page_icon="🤖"
)

# Title
st.title("🤖 Smart ML Predictor")
st.write("Data-driven prediction applications")


st.header("Welcome! 👋")

st.write("""
Welcome! This is a collection of prediction tools.

**Navigate using the sidebar →**
""")


st.info("""
These are educational demonstration tools.
""")


# About
st.header("👨‍💻 About Me")

st.write("""
Hi! I'm **Imran Butt**, learning Machine Learning.

**Skills:**
- Python Programming
- Machine Learning (Scikit-learn)
- Data Processing (Pandas, NumPy)
- Model Deployment (Streamlit)

**Contact:**
- LinkedIn: https://www.linkedin.com/in/m-imran-butt
- GitHub: https://github.com/imranbuttcodes
""")

st.write("---")


st.write("🤖 Made with Streamlit | © 2024")
