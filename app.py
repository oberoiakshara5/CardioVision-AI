
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="CardioVision AI",
    page_icon="❤️",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #101D32;
    color: #FFFFFF;
}
h1, h2, h3 {
    color: #42D6C5 !important;
}
div[data-testid="stForm"] {
    background-color: #192B42;
    padding: 25px;
    border-radius: 15px;
}
.stButton button {
    background-color: #20BFAE;
    color: #101D32;
    border-radius: 10px;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Load saved models
@st.cache_resource
def load_models():
    return joblib.load("cardiovision_models.pkl")

models = load_models()

# Header
st.title("❤️ CardioVision AI")
st.subheader("Intelligent Heart Disease Classification System")

st.markdown("""
An interactive machine-learning demonstration using
Support Vector Machine, K-Nearest Neighbors and Decision Tree.
""")

st.divider()

# Project information
col1, col2, col3 = st.columns(3)

col1.metric("ML Algorithms", "03")
col2.metric("Classification", "Binary")
col3.metric("Platform", "AI / ML")

st.divider()

st.header("Patient Information")

st.write(
    "Enter sample health-related attributes to demonstrate "
    "how the trained models generate classifications."
)

# Input form
with st.form("patient_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 18, 100, 45)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox(
            "Chest Pain Type", ["TA", "ATA", "NAP", "ASY"]
        )
        resting_bp = st.number_input(
            "Resting Blood Pressure", 0, 250, 120
        )
        cholesterol = st.number_input(
            "Cholesterol", 0, 600, 200
        )
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar", [0, 1]
        )

    with col2:
        resting_ecg = st.selectbox(
            "Resting ECG", ["Normal", "ST", "LVH"]
        )
        max_hr = st.number_input(
            "Maximum Heart Rate", 50, 250, 150
        )
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina", ["N", "Y"]
        )
        oldpeak = st.number_input(
            "Oldpeak", 0.0, 10.0, 0.0, step=0.1
        )
        st_slope = st.selectbox(
            "ST Slope", ["Up", "Flat", "Down"]
        )

    submitted = st.form_submit_button(
        "Analyse with CardioVision AI"
    )

# Prediction section
if submitted:

    patient_data = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }])

    st.divider()
    st.header("Classification Results")

    cols = st.columns(3)

    for col, (name, model) in zip(cols, models.items()):

        prediction = int(model.predict(patient_data)[0])

        with col:
            st.subheader(name)

            if prediction == 1:
                st.warning("Classified as: Heart Disease")
            else:
                st.success("Classified as: No Heart Disease")

            st.caption("Model classification output")

    st.info(
        "These are educational model outputs, not medical diagnoses. "
        "They should not be used to assess an individual's health "
        "or replace advice from a qualified healthcare professional."
    )

st.divider()

# About section
st.header("About CardioVision AI")

st.write("""
CardioVision AI is a machine-learning academic project developed
to demonstrate binary classification using a heart disease dataset.

The project implements:

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier

The models are evaluated using accuracy and F1-score.
""")

st.caption(
    "Academic project | AI & Machine Learning | CardioVision AI"
)
