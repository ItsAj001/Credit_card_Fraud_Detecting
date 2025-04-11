import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

# Page setup
st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💳 Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter transaction details to predict if it's fraudulent.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input section with expand
with st.expander("🔍 Fill in Transaction Details"):
    st.info("Please enter values for the 28 PCA features (V1–V28) and normalized transaction amount.")
    inputs = []
    cols = st.columns(3)

    for i in range(1, 29):
        col = cols[(i - 1) % 3]
        val = col.number_input(f"V{i}", value=0.0, step=0.1)
        inputs.append(val)

    norm_amount = st.number_input("💰 Normalized Amount", value=0.0, step=0.01)
    inputs.append(norm_amount)

# Prediction
st.markdown("---")
if st.button("🧠 Predict Fraud"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    st.markdown(f"**Confidence Score (Fraud):** {prediction_prob:.2f}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
