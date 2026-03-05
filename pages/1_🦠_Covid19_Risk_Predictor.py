import matplotlib.pyplot as plt
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import io


def generate_pdf_report_final(
    age, gender, comorbidity, oxygen_level, body_temperature,
    sore_throat, fatigue, headache, shortness_of_breath,
    loss_of_smell, loss_of_taste, chest_pain,
    travel_history, contact_with_patient,
    prediction, probability, model_name
):
   
    chart_data = pd.DataFrame({
        "Risk Level": ["Low Risk", "High Risk"],
        "Probability (%)": [probability[0]*100, probability[1]*100]
    })

    plt.figure(figsize=(4,3))
    plt.bar(chart_data["Risk Level"], chart_data["Probability (%)"], color=["green","red"])
    plt.title("Probability Breakdown")
    plt.ylim(0, 100)
    plt.ylabel("Probability (%)")
    plt.tight_layout()
    chart_buffer = io.BytesIO()
    plt.savefig(chart_buffer, format='png')
    chart_buffer.seek(0)
    plt.close()

    # --- Create PDF ---
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, " AI Covid Prediction Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)

    # Patient Information
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0,8, f"Model Used: {model_name}", ln=True)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 6, f"Age: {age}", ln=True)
    pdf.cell(0, 6, f"Gender: {gender}", ln=True)
    pdf.cell(0, 6, f"Comorbidity: {comorbidity}", ln=True)
    pdf.cell(0, 6, f"Oxygen Level: {oxygen_level}%", ln=True)
    pdf.cell(0, 6, f"Body Temperature: {body_temperature}°C", ln=True)
    pdf.ln(5)

    # Symptoms
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Symptoms", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 6, f"Sore Throat: {sore_throat}", ln=True)
    pdf.cell(0, 6, f"Fatigue: {fatigue}", ln=True)
    pdf.cell(0, 6, f"Headache: {headache}", ln=True)
    pdf.cell(0, 6, f"Shortness of Breath: {shortness_of_breath}", ln=True)
    pdf.cell(0, 6, f"Loss of Smell: {loss_of_smell}", ln=True)
    pdf.cell(0, 6, f"Loss of Taste: {loss_of_taste}", ln=True)
    pdf.cell(0, 6, f"Chest Pain: {chest_pain}", ln=True)
    pdf.ln(5)

    # Exposure History
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Exposure History", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 6, f"Travel History: {travel_history}", ln=True)
    pdf.cell(0, 6, f"Contact with Patient: {contact_with_patient}", ln=True)
    pdf.ln(5)

    # Prediction Result
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Prediction Result", ln=True)
    pdf.set_font("Arial", "", 12)
    risk_text = "HIGH RISK " if prediction == 1 else "LOW RISK "
    pdf.set_text_color(255,0,0) if prediction == 1 else pdf.set_text_color(0,128,0)  # red/high, green/low
    pdf.cell(0, 6, f"Risk Level: {risk_text}", ln=True)
    pdf.set_text_color(0,0,0)
    pdf.cell(0, 6, f"High Risk Probability: {probability[1]*100:.1f}%", ln=True)
    pdf.cell(0, 6, f"Low Risk Probability: {probability[0]*100:.1f}%", ln=True)
    pdf.ln(5)

    # Add probability chart
    pdf.image(chart_buffer, x=40, w=130)
    pdf.ln(5)

    #  Return PDF as bytes using dest='S' 
    pdf_bytes = pdf.output(dest='S') 
    return bytes(pdf_bytes)

# Page config
st.set_page_config(
    page_title="COVID-19 Predictor",
    page_icon="🦠"
)

# Load model
@st.cache_resource
def load_model(model_name):
    model = joblib.load(model_name)
    preprocessor = joblib.load('models/preprocessor.joblib')
    return model, preprocessor


def get_data():
    # Title
    st.title("🦠 COVID-19 Risk Predictor")
    st.write("Enter patient information to predict COVID-19 risk")

    st.write("---")
    model_choosed = st.selectbox('Choose Model', ['Logistic Regression', 'Random Forest'], index=0)
    model, preprocessor = load_model("models/corona_lg_model.joblib" if model_choosed == 'Logistic Regression' else 'models/corona_rf_model.joblib')

    # Sidebar
    if model_choosed == 'Logistic Regression':
        st.sidebar.header("Model Information")
        st.sidebar.metric("Accuracy", "76%")
        st.sidebar.metric("Recall", "89%")
        st.sidebar.metric("Precision", "72%")
    else:
        st.sidebar.header("Model Information")
        st.sidebar.metric("Accuracy", "82%")
        st.sidebar.metric("Recall", "80%")
        st.sidebar.metric("Precision", "84%")
    st.sidebar.write("---")

    st.sidebar.warning("""
    ⚠️ **Disclaimer**

    Educational tool only.
    Not for medical diagnosis.
    Consult healthcare professionals.
    """)

    st.header("Patient Information")

    # Two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Basic Details")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        
        st.subheader("Medical History")
        comorbidity = st.selectbox(
            "Pre-existing Condition", 
            ["No_Comorbidity", "Diabetes", "Asthma", "Heart Disease"]
        )
        
        st.subheader("Vital Signs")
        oxygen_level = st.slider("Oxygen Level (%)", 77, 100, 95)
        st.write(f'Oxygen Level ',oxygen_level)
        body_temperature = st.slider("Body Temperature (°C)", 36.0, 41.0, 37.0, 0.1)

    with col2:
        st.subheader("Symptoms")
        sore_throat = st.checkbox("Sore Throat")
        fatigue = st.checkbox("Fatigue")
        headache = st.checkbox("Headache")
        shortness_of_breath = st.checkbox("Shortness of Breath")
        loss_of_smell = st.checkbox("Loss of Smell")
        loss_of_taste = st.checkbox("Loss of Taste")
        chest_pain = st.checkbox("Chest Pain")
        
        st.subheader("Exposure History")
        travel_history = st.checkbox("Recent Travel (14 days)")
        contact_with_patient = st.checkbox("Contact with COVID Patient")
    st.write("---")
    return model, preprocessor, model_choosed, {
        'age': age,
        'gender': gender,
        'sore_throat': int(sore_throat),
        'fatigue': int(fatigue),
        'headache': int(headache),
        'shortness_of_breath': int(shortness_of_breath),
        'loss_of_smell': int(loss_of_smell),
        'loss_of_taste': int(loss_of_taste),
        'oxygen_level': oxygen_level,
        'body_temperature': body_temperature,
        'comorbidity': comorbidity,
        'travel_history': int(travel_history),
        'contact_with_patient': int(contact_with_patient),
        'chest_pain': int(chest_pain)
    }

def prepareData(input_data,preprocessor):
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical variables
    input_df = pd.get_dummies(input_df, columns=['comorbidity'], prefix='comorbidity')
    input_df = pd.get_dummies(input_df, columns=['gender'], prefix='gender')
    
    # Add missing columns with 0
    expected_columns = preprocessor.feature_names_in_
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match training
    input_df = input_df[expected_columns]
    
    # Apply preprocessing (scaling)
    input_scaled = preprocessor.transform(input_df)
    
    # Convert to DataFrame with proper column names
    return pd.DataFrame(
        input_scaled, 
        columns=preprocessor.get_feature_names_out()
    )




def main():
    model, preprocessor, model_name, input_data = get_data()

    # Predict button
    if st.button("🔍 Predict Risk", type="primary"):
        # Create DataFrame
        input_scaled_df = prepareData(input_data, preprocessor)
        
        # Make prediction
        prediction = model.predict(input_scaled_df)[0]
        probability = model.predict_proba(input_scaled_df)[0]
        
        # Display results
        st.write("---")
        st.header("📋 Prediction Results")
        
        if prediction == 1:
            st.error("### ⚠️ HIGH RISK")
            st.metric("Confidence", f"{probability[1]*100:.1f}%")
            
            st.warning("""
            **Recommended Actions:**
            
            - 🏠 Self-isolate immediately
            - 🧪 Get tested (PCR/Rapid test)
            - 👨‍⚕️ Contact healthcare provider
            - 📞 Inform close contacts
            - 🌡️ Monitor symptoms closely
            """)
            
        else:
            st.success("### ✅ LOW RISK")
            st.metric("Confidence", f"{probability[0]*100:.1f}%")
            
            st.info("""
            **Recommendations:**
            
            - 👀 Continue monitoring symptoms
            - 🧪 Get tested if symptoms worsen
            - 😷 Maintain safety precautions
            - 💧 Stay hydrated and rest
            - 👨‍⚕️ Consult doctor if concerned
            """)
        
        # Probability breakdown
        st.write("---")
        st.subheader("Probability Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("High Risk Probability", f"{probability[1]*100:.1f}%")
        with col2:
            st.metric("Low Risk Probability", f"{probability[0]*100:.1f}%")
        
        # Simple bar chart
        chart_data = pd.DataFrame({
        "Risk Level": ["Low Risk", "High Risk"],
        "Probability (%)": [probability[0]*100, probability[1]*100]
        })

        st.bar_chart(chart_data.set_index("Risk Level"))

        pdf_bytes = generate_pdf_report_final(
        input_data['age'], input_data['gender'], input_data["comorbidity"], input_data["oxygen_level"], input_data["body_temperature"],
        input_data["sore_throat"], input_data["fatigue"], input_data['headache'], input_data['shortness_of_breath'],
        input_data['loss_of_smell'], input_data['loss_of_taste'], input_data['chest_pain'],
        input_data['travel_history'], input_data['contact_with_patient'],
        prediction, probability, model_name
        )

        st.download_button(
            label="💾 Download Your Report",
            data=pdf_bytes,
            file_name="covid_report.pdf",
            mime="application/pdf"
        )

    st.write("Educational demonstration tool")

if __name__ == '__main__':
    main()