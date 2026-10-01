import streamlit as st
import pandas as pd
import joblib
import base64


model = joblib.load("logistic_regression.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Early Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Early Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

with open("healthcare.jpg", "rb") as f:
    bg_image = base64.b64encode(f.read()).decode()

# 🎨 Custom styling
st.markdown(f"""
<style>

/* ===== MAIN TITLE ===== */
h1 {{
    text-align: center;
    color: #f4f1f8;
    font-size: 2.7rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}}

/* ===== SUBTITLE ===== */
.subtitle {{
    text-align: center;
    color: #ddd9e5;
    font-size: 1.4rem;
    font-weight: 500;
    margin-top: 0.3rem;
    margin-bottom: 1rem;
}}

/* ===== PATIENT INFORMATION ===== */
h3 {{
    text-align: left;
    color: #e8e4f0;
}}

/* ===== INTRO BOX ===== */
.intro-box {{
    background-color: #181a24;
    border: 1px solid #3b3d4d;
    border-left: 4px solid #7567c7;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0 25px 0;
    color: #d8d5df;
    font-size: 15px;
}}

/* ===== PREDICT BUTTON ===== */
div.stButton > button {{
    background-color: #5146A8;
    color: #F1EFF8;
    border: 1px solid #6559BD;
    border-radius: 8px;
    padding: 0.55rem 1rem;
    font-weight: 600;
    transition: 0.2s;
}}

/* ===== BUTTON HOVER ===== */
div.stButton > button:hover {{
    background-color: #5C50B5;
    border-color: #7569CA;
    color: white;
}}

/* ===== MAIN BACKGROUND ===== */
.stApp {{
    background-color: #080a0f;
}}

/* ===== HEALTHCARE BACKGROUND IMAGE ===== */
.stApp::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 500px;

    background:
    linear-gradient(
        to bottom,
        rgba(8, 10, 15, 0.35) 0%,
        rgba(8, 10, 15, 0.55) 40%,
        rgba(8, 10, 15, 0.85) 75%,
        #080a0f 100%
    ),
    url("data:image/png;base64,{bg_image}");

    background-size: cover;
    background-position: center top;
    background-repeat: no-repeat;

    z-index: 0;
    pointer-events: none;
}}

/* ===== INFO MESSAGE ===== */
div[data-testid="stAlert"] {{
    background-color: #191b25;
    border: 1px solid #3b3d4d;
    border-left: 4px solid #7567c7;
    border-radius: 8px;
    color: #d8d5df;
}}

</style>
""", unsafe_allow_html=True)

model = joblib.load("logistic_regression.pkl")
scaler = joblib.load("scaler.pkl")



st.title(" Early Disease Prediction")

st.markdown(
    '<div class="subtitle">Machine learning-based health risk assessment</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="intro-box">
    Enter the patient's details below.
</div>
""", unsafe_allow_html=True)

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45
    )

    bmi = st.number_input(
        "BMI",
        min_value=1.0,
        value=27.5
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=1.0,
        value=200.0
    )

    heart_rate = st.number_input(
        "Heart Rate (bpm)",
        min_value=1.0,
        value=75.0
    )

    exercise = st.selectbox(
        "Exercise Level",
        ["Low", "Moderate", "High"]
    )

with col2:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)",
        min_value=1.0,
        value=120.0
    )

    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=1.0,
        value=100.0
    )

    smoking = st.selectbox(
        "Smoking",
        ["No", "Yes"]
    )


# Convert categorical values to numbers
gender_value = 0 if gender == "Female" else 1
smoking_value = 0 if smoking == "No" else 1

exercise_value = {
    "Low": 0,
    "Moderate": 1,
    "High": 2
}[exercise]


st.divider()

if st.button("🔍 Predict Disease", use_container_width=True):

    new_patient = pd.DataFrame({
        "Age": [age],
        "Gender": [gender_value],
        "BMI": [bmi],
        "Blood_Pressure_mmHg": [blood_pressure],
        "Cholesterol_mg_dL": [cholesterol],
        "Glucose_mg_dL": [glucose],
        "Heart_Rate_bpm": [heart_rate],
        "Smoking": [smoking_value],
        "Exercise_Level": [exercise_value]
    })

    # Scale the input
    new_patient_scaled = scaler.transform(new_patient)

    # Predict
    prediction = model.predict(new_patient_scaled)

    st.subheader("🔎 Prediction Result")

    if prediction[0] == "Disease":
        st.error("⚠️ Disease Predicted")
    else:
        st.success("✅ No Disease Predicted")

    st.caption("Based on the health information entered.")

