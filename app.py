import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Weather Rain Prediction", layout="centered")
st.title("🌧️ Rain Prediction App")
st.write("Predict whether it will rain tomorrow based on weather conditions")

# Load models
models = {
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "Decision Tree": joblib.load("decision_tree.pkl"),
    "Random Forest": joblib.load("random_forest.pkl")
}

# Load scaler and label encoders
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Model selection
selected_model_name = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# Prepare input fields
inputs = {}
for col, le in label_encoders.items():
    if col != "RainTomorrow":
        inputs[col] = st.selectbox(col, le.classes_)

numeric_features = [
    "MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine",
    "WindGustSpeed","WindSpeed9am","WindSpeed3pm",
    "Humidity9am","Humidity3pm","Pressure9am","Pressure3pm",
    "Cloud9am","Cloud3pm","Temp9am","Temp3pm"
]

for col in numeric_features:
    if col not in inputs:
        inputs[col] = st.number_input(col, value=0.0)

# Convert input to DataFrame
input_df = pd.DataFrame([inputs])

# Encode categorical features
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][prediction]
    else:
        # Decision tree may not have predict_proba
        prob = 1.0
    
    if prediction == 1:
        st.error(f"🌧️ Rain Tomorrow (Confidence: {prob:.2f})")
    else:
        st.success(f"☀️ No Rain Tomorrow (Confidence: {prob:.2f})")

# Footer
st.markdown(
    "<div style='text-align:center; margin-top:40px;'>Created by Abdul Raheem Liaqat</div>",
    unsafe_allow_html=True
)
