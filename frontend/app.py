import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import os

st.set_page_config(page_title="LKR AI Forecaster", layout="wide")

# Backend API URL (Use "localhost" for local testing, or service name for Docker)
BACKEND_URL = "http://localhost:8000/predict"

# Load data locally for the charts
@st.cache_data
def load_data():
    # Adjusted to find data from the frontend folder
    # Assuming: Project_Root/frontend/app.py
    data_path = os.path.join('..', 'data', 'processed', 'LKR_Forex_Processed.csv')
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

st.title("🌍 LKR Exchange Rate AI Forecaster")
selected_curr = st.sidebar.selectbox("Select Currency", df['Currency'].unique())

# Prepare the latest data to send to the API
curr_df = df[df['Currency'] == selected_curr].sort_values('Date')
latest_row = curr_df.iloc[-1].to_dict()

# Fix for JSON serialization (convert Timestamp to string)
latest_row['Date'] = str(latest_row['Date'])

if st.button("Generate Forecast"):
    with st.spinner("Requesting AI Prediction from Backend..."):
        response = requests.post(BACKEND_URL, json={
            "currency": selected_curr,
            "latest_data": latest_row
        })
        
        if response.status_code == 200:
            result = response.json()
            pred = result['predicted_price']
            curr_price = latest_row['LKR_Rate']
            
            # Display Results
            col1, col2 = st.columns(2)
            col1.metric("Current Rate", f"Rs. {curr_price:.2f}")
            col2.metric("AI Forecast", f"Rs. {pred:.2f}", f"{pred-curr_price:+.2f}")
            
            # Chart logic here...
        else:
            st.error("Could not connect to the Backend API.")