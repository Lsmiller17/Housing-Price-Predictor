import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# URL of the model file in GitHub
MODEL_URL = "https://raw.githubusercontent.com/Lsmiller17/Housing-Price-Predictor/main/Housing_Price_Model_Scikit_Learn_1_6_1.pkl"
MODEL_PATH = "Housing_Price_Model_Scikit_Learn_1_6_1.pkl"

# Function to download the model if it's missing
def download_model():
    if not os.path.exists(MODEL_PATH):
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
        else:
            raise FileNotFoundError("⚠️ Unable to download the model. Ensure the GitHub link is correct.")

# Ensure the model file is available
download_model()

# Load the trained model
model = joblib.load(MODEL_PATH)

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
