# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page config
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for hospital-like theme
st.markdown("""
<style>
    /* Import Font Awesome for icons */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
    
    /* Overall background and text */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Header styling */
    .hospital-header {
        background: linear-gradient(135deg, #0b4d8a 0%, #1e6f9f 100%);
        padding: 1.5rem;
        border-radius: 0 0 20px 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .hospital-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .hospital-header p {
        margin: 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styling for input sections */
    .input-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .input-card:hover {
        box-shadow: 0 6px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Section titles with icons */
    .section-title {
        color: #0b4d8a;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    .section-title i {
        margin-right: 0.5rem;
        color: #1e6f9f;
    }
    
    /* Result box styling */
    .result-box {
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-top: 2rem;
        background: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 8px solid;
    }
    .result-low {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #f0fdf4, #ffffff);
    }
    .result-high {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2, #ffffff);
    }
    .result-text {
        font-size: 2rem;
        font-weight: 700;
    }
    .probability-bar {
        height: 10px;
        border-radius: 5px;
        background: #e2e8f0;
        margin: 1rem 0;
    }
    .probability-fill {
        height: 10px;
        border-radius: 5px;
        background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f1f5f9;
    }
    .sidebar-info {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Streamlit element overrides */
    .stButton > button {
        background-color: #0b4d8a;
        color: white;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #1e6f9f;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: scale(1.02);
    }
    .stSelectbox, .stNumberInput {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Header
st.markdown("""
<div class="hospital-header">
    <h1><i class="fas fa-heartbeat"></i> Heart Disease Risk Assessment</h1>
    <p>Advanced KNN-based prediction tool for early detection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.markdown("<div class='sidebar-info'><h3><i class='fas fa-info-circle'></i> About</h3><p>This tool uses a K-Nearest Neighbors (KNN) machine learning model trained on the Cleveland Heart Disease dataset. It predicts the likelihood of heart disease based on medical attributes.</p></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-info'><h3><i class='fas fa-database'></i> Dataset</h3><p><strong>Source:</strong> UCI Machine Learning Repository<br><strong>Samples:</strong> 303<br><strong>Features:</strong> 13<br><strong>Accuracy:</strong> ~85%</p></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-info'><h3><i class='fas fa-stethoscope'></i> Instructions</h3><p>Fill in all patient details below. The model will analyze the input and return a risk assessment with probability.</p></div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-info'><h3><i class='fas fa-shield-alt'></i> Disclaimer</h3><p>This is for educational purposes only. Always consult a healthcare professional.</p></div>", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Patient information section
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><i class='fas fa-user'></i> Patient Demographics</div>", unsafe_allow_html=True)
    
    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50, help="Patient's age in years")
    with demo_col2:
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Clinical measurements
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><i class='fas fa-heart'></i> Clinical Measurements</div>", unsafe_allow_html=True)
    
    meas_col1, meas_col2, meas_col3 = st.columns(3)
    with meas_col1:
        trestbps = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
        thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    with meas_col2:
        cp = st.selectbox("Chest Pain Type", options=[0,1,2,3], 
                          format_func=lambda x: ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"][x])
        fbs = st.selectbox("Fasting Blood Sugar >120", options=[0,1], format_func=lambda x: "False" if x==0 else "True")
        exang = st.selectbox("Exercise Induced Angina", options=[0,1], format_func=lambda x: "No" if x==0 else "Yes")
    with meas_col3:
        restecg = st.selectbox("Resting ECG", options=[0,1,2],
                               format_func=lambda x: ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"][x])
        oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of ST", options=[0,1,2],
                             format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Additional factors
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'><i class='fas fa-flask'></i> Additional Factors</div>", unsafe_allow_html=True)
    
    add_col1, add_col2, add_col3 = st.columns(3)
    with add_col1:
        ca = st.selectbox("Major Vessels (0-3)", options=[0,1,2,3])
    with add_col2:
        thal = st.selectbox("Thalassemia", options=[1,2,3],
                            format_func=lambda x: ["Normal", "Fixed defect", "Reversible defect"][x-1])
    with add_col3:
        # Placeholder for any extra? Not needed
        st.write("")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Predict button centered
    col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
    with col_btn2:
        predict_button = st.button("🔍 ASSESS RISK", use_container_width=True)

with col2:
    # Display a medical illustration or placeholder
    st.image("https://img.icons8.com/fluency/96/null/heart-with-pulse.png", width=100)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Why KNN?")
    st.info("K-Nearest Neighbors is a simple yet effective algorithm for medical diagnosis. It compares patient data with similar historical cases to make predictions.")
    
    # Show feature importance (static)
    st.markdown("### Key Factors")
    st.markdown("""
    - **Chest pain type**  
    - **Max heart rate**  
    - **ST depression**  
    - **Number of vessels**  
    """)

# Prediction and results
if predict_button:
    # Prepare input array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    # Scale
    features_scaled = scaler.transform(features)
    # Predict
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]
    
    # Display result in a nice box
    st.markdown("---")
    if prediction == 1:
        result_class = "result-high"
        result_icon = "⚠️"
        result_title = "High Risk"
        result_color = "#ef4444"
    else:
        result_class = "result-low"
        result_icon = "✅"
        result_title = "Low Risk"
        result_color = "#10b981"
    
    st.markdown(f"""
    <div class='result-box {result_class}'>
        <span style='font-size:3rem;'>{result_icon}</span>
        <div class='result-text' style='color:{result_color};'>{result_title}</div>
        <p style='font-size:1.2rem;'>Probability of heart disease: <strong>{prob[1]:.2%}</strong></p>
        <div class='probability-bar'>
            <div class='probability-fill' style='width:{prob[1]*100}%;'></div>
        </div>
        <p style='color:#64748b;'>This assessment is based on a KNN model with 85% accuracy. Always consult a doctor.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Optionally show raw probabilities
    with st.expander("Show detailed probabilities"):
        st.write(f"**No Disease:** {prob[0]:.2%}")
        st.write(f"**Disease:** {prob[1]:.2%}")

# Footer
st.markdown("""
<div class="footer">
    <i class="fas fa-code-branch"></i> Developed with ❤️ using Streamlit | Data source: UCI ML Repository
</div>
""", unsafe_allow_html=True)
