import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
# Load model
model = load_model("my_model.keras")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction")

st.write("Enter patient information:")

# Inputs (حسب أعمدة Heart Disease Dataset الشائعة)
age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, value=0)

trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)

chol = st.number_input("Cholesterol", min_value=50, max_value=600, value=200)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120",
    options=[0, 1]
)

restecg = st.number_input("Resting ECG", min_value=0, max_value=2, value=0)

thalach = st.number_input("Maximum Heart Rate", min_value=50, max_value=250, value=150)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1]
)

oldpeak = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

slope = st.number_input("Slope", min_value=0, max_value=2, value=0)

ca = st.number_input("Number of Major Vessels", min_value=0, max_value=4, value=0)

thal = st.number_input("Thal", min_value=0, max_value=3, value=0)


# Prediction
if st.button("Predict"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

   
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0][0] > .5:
        st.error("✅ No Heart Disease detected")
    else:
        st.success("⚠️ High risk of Heart Disease")