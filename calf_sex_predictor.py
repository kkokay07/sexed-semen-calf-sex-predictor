import pandas as pd
import streamlit as st
import joblib

# Load trained model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Load dataset for dropdowns
data = pd.read_csv("data_cs.txt", sep="\t")

st.title("🐮 Calf Sex Prediction Tool")
st.write("Select input features to predict calf sex (Male/Female).")

# Input selections
breed = st.selectbox("Select Breed", sorted(data["Breed"].unique()))
parity = st.selectbox("Select Parity", sorted(data["Parity"].unique()))
mgmt = st.selectbox("Select Management System", sorted(data["Management_Systems"].unique()))

if st.button("🔍 Predict Calf Sex"):
    input_df = pd.DataFrame({
        'Breed': [breed],
        'Parity': [parity],
        'Management_Systems': [mgmt]
    })
    input_encoded = pd.DataFrame({
        col: [encoders[col].transform(input_df[col])[0]]
        for col in input_df.columns
    })
    pred = model.predict(input_encoded)[0]
    sex = target_encoder.inverse_transform([pred])[0]
    st.success(f"🧬 Predicted Calf Sex: **{sex}**")
