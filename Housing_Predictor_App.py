import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# GitHub raw file URL for the model (update the repo name accordingly)
MODEL_URL = "https://raw.githubusercontent.com/lsmiller17/Housing-Price-Predictor/main/Housing_Price_Model.pkl"

# Function to download the model if it's missing
def download_model():
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open("Housing_Price_Model.pkl", "wb") as f:
            f.write(response.content)
    else:
        st.error("⚠️ Unable to download the model. Ensure the GitHub link is correct.")
        st.stop()

# Ensure model exists
if not os.path.exists("Housing_Price_Model.pkl"):
    download_model()

# Load the trained model
model = joblib.load("Housing_Price_Model.pkl")

# Define prediction function
def predict_price(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]

# Streamlit UI
st.title('Ames Housing Price Predictor')

input_data = {
    'GrLivArea': st.number_input('Above Ground Living Area (sq ft)', min_value=500, max_value=5000, value=1500),
    'GarageCars': st.slider('Number of Garage Spaces', min_value=0, max_value=4, value=1),
    'TotalBsmtSF': st.number_input('Total Basement Area (sq ft)', min_value=0, max_value=3000, value=800),
    'YearBuilt': st.number_input('Year Built', min_value=1800, max_value=2024, value=2000)
}

if st.button('Predict Price'):
    price = predict_price(input_data)
    st.write(f'🏡 **Predicted House Price: ${price:,.2f}**')
