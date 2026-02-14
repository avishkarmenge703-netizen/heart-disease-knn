# app.py
import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction using KNN")
st.write("Enter patient details to predict the likelihood of heart disease.")

# Input fields (same order as training)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type", options=[0,1,2,3], 
                      format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0,1], format_func=lambda x: "False" if x==0 else "True")
    restecg = st.selectbox("Resting ECG Results", options=[0,1,2],
                           format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", options=[0,1,2],
                         format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    ca = st.selectbox("Number of Major Vessels (0-3)", options=[0,1,2,3])
    thal = st.selectbox("Thalassemia", options=[1,2,3],
                        format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect"][x-1])

# Predict button
if st.button("Predict"):
    # Prepare input array (must match training feature order)
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    # Scale
    features_scaled = scaler.transform(features)
    # Predict
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]

    if prediction == 1:
        st.error(f"⚠️ **High risk** of heart disease (probability: {prob[1]:.2f})")
    else:
        st.success(f"✅ **Low risk** of heart disease (probability: {prob[0]:.2f})")
