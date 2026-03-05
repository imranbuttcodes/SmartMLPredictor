import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF
import io
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="💊"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('models/diabetes_model.joblib')
    scaler = joblib.load('models/diabetes_scaler.joblib')
    return model, scaler




# Function to create PDF
def create_pdf(input_data, prediction, probability):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Diabetes Prediction Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 8, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Model Used: Logistic Regression", ln=True)
    pdf.ln(5)
    
    # Patient Data
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Patient Medical Data:", ln=True)
    for key, value in input_data.items():
        pdf.cell(0, 8, f"{key}: {value}", ln=True)
    
    pdf.ln(5)
    pdf.cell(0, 10, f"Prediction: {'HIGH RISK - Diabetic' if prediction==1 else 'LOW RISK - Non-Diabetic'}", ln=True)
    pdf.cell(0, 8, f"Confidence: {probability[prediction]*100:.1f}%", ln=True)
    
    pdf.ln(5)
    pdf.cell(0, 10, "Recommendations:", ln=True)
    
    if prediction == 1:
        recommendations = [
            "Consult an endocrinologist",
            "Get HbA1c test",
            "Start dietary changes",
            "Regular exercise routine",
            "Monitor blood sugar regularly",
            "Discuss medication options"
        ]
    else:
        recommendations = [
            "Maintain healthy diet",
            "Stay physically active",
            "Monitor weight regularly",
            "Regular health checkups",
            "Avoid high sugar intake",
            "Get adequate sleep"
        ]
    
    for rec in recommendations:
        pdf.cell(0, 8, f"- {rec}", ln=True)
    
    # Add probability bar chart
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(['Non-Diabetic', 'Diabetic'], [probability[0]*100, probability[1]*100], color=['green','red'])
    ax.set_ylabel("Probability (%)")
    ax.set_title("Prediction Probability")
    
    # Save chart to in-memory image
    chart_buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(chart_buffer, format='PNG')
    plt.close(fig)
    chart_buffer.seek(0)
    
    # Insert image into PDF
    pdf.image(chart_buffer, x=50, w=110)  # adjust x and width as needed
    
    # Save PDF to BytesIO
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# Add download button after prediction results


model, scaler = load_model()

# Title
st.title("💊 Diabetes Risk Predictor")
st.write("Enter patient medical information to predict diabetes risk")

st.write("---")

# Sidebar
st.sidebar.write('**Model Used:** Logistic Regression')
st.sidebar.header("Model Information")
st.sidebar.metric("Accuracy", "73.38%")
st.sidebar.metric("Precision", "59.42%")
st.sidebar.metric("Recall", "75.93%")
st.sidebar.metric("F1 Score", "67%")

st.sidebar.write("---")

st.sidebar.warning("""
⚠️ **Disclaimer**

Educational tool only.
Not for medical diagnosis.
Consult healthcare professionals.
""")

# Input form
st.header("Patient Medical Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    
    st.subheader("Glucose & BMI")
    glucose = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
    bmi = st.slider("BMI (Body Mass Index)", 0.0, 70.0, 25.0, 0.1)

with col2:
    st.subheader("Family History")
    diabetes_pedigree = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
    
    st.subheader("Additional Info")
    st.write("These features are auto-calculated:")
    st.info(f"High Glucose: {'Yes' if glucose > 140 else 'No'}")
    st.info(f"High BMI: {'Yes' if bmi > 30 else 'No'}")

st.write("---")

# Predict button
if st.button("🔍 Predict Diabetes Risk", type="primary"):
    
    # Feature engineering (same as training!)
    high_glucose = 1 if glucose > 140 else 0
    high_bmi = 1 if bmi > 30 else 0
    bmi_glucose = bmi * glucose
    age_glucose = age * glucose
    
    # Prepare input data (match training features!)
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age,
        'HighGlucose': high_glucose,
        'HighBMI': high_bmi,
        'BMI_Glucose': bmi_glucose,
        'Age_Glucose': age_glucose
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Scale features (same order as training!)
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display results
    st.write("---")
    st.header("📋 Prediction Results")
    
    if prediction == 1:
        st.error("### ⚠️ HIGH RISK - Diabetic")
        st.metric("Confidence", f"{probability[1]*100:.1f}%")
        
        st.warning("""
        **Recommended Actions:**
        
        - 🏥 Consult an endocrinologist
        - 🧪 Get HbA1c test
        - 🍎 Start dietary changes
        - 🏃 Regular exercise routine
        - 📊 Monitor blood sugar regularly
        - 💊 Discuss medication options
        """)
        
    else:
        st.success("### ✅ LOW RISK - Non-Diabetic")
        st.metric("Confidence", f"{probability[0]*100:.1f}%")
        
        st.info("""
        **Recommendations:**
        
        - 🍎 Maintain healthy diet
        - 🏃 Stay physically active
        - ⚖️ Monitor weight regularly
        - 🧪 Regular health checkups
        - 🚫 Avoid high sugar intake
        - 😴 Get adequate sleep
        """)
    
    # Probability breakdown
    st.write("---")
    st.subheader("Probability Breakdown")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Diabetic Probability", f"{probability[1]*100:.1f}%")
    with col2:
        st.metric("Non-Diabetic Probability", f"{probability[0]*100:.1f}%")

    
    # Simple bar chart
    st.bar_chart({
        'Non-Diabetic': probability[0]*100,
        'Diabetic': probability[1]*100
    })
    
    # Show input summary
    with st.expander("🔍 View Input Summary"):
        st.write("**Patient Data Entered:**")
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown(f"""
            **Basic Info:**
            - Age: {age} years
            - Pregnancies: {pregnancies}
            
            **Medical Metrics:**
            - Glucose: {glucose} mg/dL
            - BMI: {bmi}
            - Pedigree: {diabetes_pedigree:.3f}
            """)
        
        with summary_col2:
            st.markdown(f"""
            **Engineered Features:**
            - High Glucose: {'Yes' if high_glucose else 'No'}
            - High BMI: {'Yes' if high_bmi else 'No'}
            - BMI x Glucose: {bmi_glucose:.2f}
            - Age x Glucose: {age_glucose:.2f}
            """)

        
    pdf_file = create_pdf(input_data, prediction, probability)
    st.download_button(
            label="💾 Download PDF Report",
            data=pdf_file,
            file_name="diabetes_prediction_report.pdf",
            mime="application/pdf"
    )   


# Footer
st.write("---")
st.write("Educational demonstration tool")
