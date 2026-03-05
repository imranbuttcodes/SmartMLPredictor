import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import io

# --- Load Models and Transformers ---
rf_model = joblib.load("models/rf_titanic_model.joblib")
lg_model = joblib.load("models/lg_titanic_model.joblib")
dt_model = joblib.load("models/dt_titanic_model.joblib")
knn_model = joblib.load("models/knn_titanic_model.joblib")
encoder = joblib.load("models/titanic_encoder.joblib")
scaler = joblib.load("models/titanic_scaler.joblib")  # double check spelling

# --- Page Config ---
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢"
)

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival probability.")

# --- User Input ---
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    family_size = st.number_input("Family Size (SibSp + Parch)", min_value=0, max_value=10, value=1)

with col2:
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# --- Prepare Input Data ---
input_data = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex,
    "Age": age,
    "FamilySize": family_size,
    "Embarked": embarked
}])

num_cols = ["Pclass", "Age", "FamilySize"]
cat_cols = ["Sex", "Embarked"]

input_data[num_cols] = input_data[num_cols].astype("float64")
input_num = scaler.transform(input_data[num_cols])
input_cat = encoder.transform(input_data[cat_cols])
input_final = np.hstack([input_cat, input_num])

# --- Model Selection ---
model_choice = st.radio("Choose Model", ["Random Forest", "Logistic Regression", "Decision Tree", "KNN"])

# --- Sidebar Model Performance Info ---
st.sidebar.title("📊 Model Performance")

model_metrics = {
    "KNN": {
        "Accuracy": "0.79",
        "Precision": "0.83",
        "Recall": "0.56",
        "F1-Score": "0.67"
    },
    "Logistic Regression": {
        "Accuracy": "0.79",
        "Precision": "0.70",
        "Recall": "0.78",
        "F1-Score": "0.74"
    },
    "Decision Tree": {
        "Accuracy": "0.78",
        "Precision": "0.81",
        "Recall": "0.55",
        "F1-Score": "0.65"
    },
    "Random Forest": {
        "Accuracy": "0.78",
        "Precision": "0.80",
        "Recall": "0.56",
        "F1-Score": "0.66"
    }
}

selected_metrics = model_metrics[model_choice]

st.sidebar.subheader(f"🔎 {model_choice} Metrics")
st.sidebar.write(f"**Accuracy:** {selected_metrics['Accuracy']}")
st.sidebar.write(f"**Precision:** {selected_metrics['Precision']}")
st.sidebar.write(f"**Recall:** {selected_metrics['Recall']}")
st.sidebar.write(f"**F1-Score:** {selected_metrics['F1-Score']}")



def generate_pdf(input_data, prediction, probability, model_name):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0,10,"Titanic Survival Report", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 12)
    pdf.cell(0,8,f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(0,8,f"Model Used: {model_name}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0,8,"Passenger Info", ln=True)
    pdf.set_font("Arial", "", 12)
    for col in input_data.columns:
        pdf.cell(0,6,f"{col}: {input_data[col][0]}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0,8,"Prediction", ln=True)
    pdf.set_font("Arial", "", 12)
    status = "Survived " if prediction == 1 else "Did Not Survive "
    pdf.cell(0,6,f"Prediction: {status}", ln=True)
    pdf.cell(0,6,f"Survival Probability: {probability[1]*100:.1f}%", ln=True)
    pdf.cell(0,6,f"Did Not Survive Probability: {probability[0]*100:.1f}%", ln=True)
    pdf.ln(5)
    
    # Add bar chart
    chart_data = pd.DataFrame({
        "Outcome": ["Did Not Survive", "Survived"],
        "Probability (%)": [probability[0]*100, probability[1]*100]
    })
    plt.figure(figsize=(4,3))
    plt.bar(chart_data["Outcome"], chart_data["Probability (%)"], color=["red","green"])
    plt.ylim(0,100)
    plt.tight_layout()
    chart_buffer = io.BytesIO()
    plt.savefig(chart_buffer, format='png')
    chart_buffer.seek(0)
    plt.close()
    
    pdf.image(chart_buffer, x=40, w=130)
    
    pdf_bytes = pdf.output(dest='S')
    return bytes(pdf_bytes)

# --- Prediction ---
if st.button("Predict Survival"):
    model = {
        "Random Forest": rf_model,
        "Logistic Regression": lg_model,
        "Decision Tree": dt_model,
        "KNN": knn_model
    }[model_choice]
    
    prediction = model.predict(input_final)[0]
    probability = model.predict_proba(input_final)[0]
    
    st.write("---")
    if prediction == 1:
        st.success("Passenger would **Survive**!")
    else:
        st.error("Passenger would **Not Survive**!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Did Not Survive Probability", f"{probability[0]*100:.1f}%")
    with col2:
        st.metric("Survival Probability", f"{probability[1]*100:.1f}%")
    
    # Bar chart
    chart_data = pd.DataFrame({
        "Outcome": ["Did Not Survive", "Survived"],
        "Probability (%)": [probability[0]*100, probability[1]*100]
    })
    st.bar_chart(chart_data.set_index("Outcome"))
    # PDF download
    pdf_bytes = generate_pdf(input_data, prediction, probability, model_choice)
    st.download_button(
        label="💾 Download Report",
        data=pdf_bytes,
        file_name="titanic_report.pdf",
        mime="application/pdf"
    )
    
st.write("Educational tool only. Not for actual decision making.")

