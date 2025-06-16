import pandas as pd
import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("best_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Load dataset for dropdown options
data = pd.read_csv("data_cs.txt", sep="\t")

st.title("🐮 Calf Sex Prediction Tool")
st.write("Select input features to predict calf sex (Male/Female).")

# Dropdown inputs
breed = st.selectbox("Select Breed", sorted(data["Breed"].unique()))
parity = st.selectbox("Select Parity", sorted(data["Parity"].unique()))
mgmt = st.selectbox("Select Management System", sorted(data["Management_Systems"].unique()))
semen_type = st.selectbox("Select Semen Type", sorted(data["SEMEN_TYPE"].unique()))

# Prediction button
if st.button("🔍 Predict Calf Sex"):
    input_df = pd.DataFrame({
        'Breed': [breed],
        'Parity': [parity],
        'Management_Systems': [mgmt],
        'SEMEN_TYPE': [semen_type]
    })

    # Encode the input
    input_encoded = pd.DataFrame({
        col: [encoders[col].transform(input_df[col])[0]]
        for col in input_df.columns
    })

    # Predict and decode
    pred = model.predict(input_encoded)[0]
    sex = target_encoder.inverse_transform([pred])[0]

    st.success(f"🧬 Predicted Calf Sex: **{sex}**")
