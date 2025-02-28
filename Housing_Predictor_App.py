import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Train the model inside Streamlit to ensure compatibility
def train_model():
    # Generate dataset
    df = pd.DataFrame({
        'GrLivArea': np.random.randint(500, 5000, 1000),
        'GarageCars': np.random.randint(0, 4, 1000),
        'TotalBsmtSF': np.random.randint(0, 3000, 1000),
        'YearBuilt': np.random.randint(1800, 2024, 1000),
        'SalePrice': np.random.randint(50000, 500000, 1000)
    })

    # Select features
    X = df[['GrLivArea', 'GarageCars', 'TotalBsmtSF', 'YearBuilt']]
    y = df['SalePrice']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

# Train the model in the deployment environment
model = train_model()

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
