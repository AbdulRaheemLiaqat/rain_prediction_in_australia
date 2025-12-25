import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Weather Rain Prediction", layout="centered")
st.title("🌧️ Rain Prediction App")
st.write("Predict whether it will rain tomorrow based on weather conditions")

models = {
    "Logistic Regression": joblib.load("logistic_regression.pkl"),
    "Decision Tree": joblib.load("decision_tree.pkl"),
    "Random Forest": joblib.load("random_forest.pkl")
}

scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

selected_model_name = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

feature_order = [
    "Date", "Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation",
    "Sunshine", "WindGustDir", "WindGustSpeed", "WindDir9am",
    "WindDir3pm", "WindSpeed9am", "WindSpeed3pm", "Humidity9am",
    "Humidity3pm", "Pressure9am", "Pressure3pm", "Cloud9am",
    "Cloud3pm", "Temp9am", "Temp3pm", "RainToday"
]

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
    inputs[col] = st.number_input(col, value=0.0)

input_df = pd.DataFrame([inputs])
input_df = input_df.reindex(columns=feature_order)

for col in numeric_features:
    if col in input_df.columns:
        input_df[col] = input_df[col].astype(float)

for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

input_scaled = scaler.transform(input_df.values)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_scaled)[0][prediction]
    else:
        prob = 1.0

    if prediction == 1:
        st.error(f"🌧️ Rain Tomorrow (Confidence: {prob:.2f})")
    else:
        st.success(f"☀️ No Rain Tomorrow (Confidence: {prob:.2f})")

st.markdown(
    "<div style='text-align:center; margin-top:40px;'>Created by Abdul Raheem Liaqat</div>",
    unsafe_allow_html=True
)
