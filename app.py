import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Weather Rain Prediction", layout="centered")

with open("random_forest.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

st.title("🌧️ Rain Prediction App")
st.write("Predict whether it will rain tomorrow based on weather conditions")

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

input_df = pd.DataFrame([inputs])

for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction]

    if prediction == 1:
        st.error(f"🌧️ Rain Tomorrow (Confidence: {prob:.2f})")
    else:
        st.success(f"☀️ No Rain Tomorrow (Confidence: {prob:.2f})")

st.markdown(
    "<div style='text-align:center; margin-top:40px;'>Created by Abdul Raheem Liaqat</div>",
    unsafe_allow_html=True
)