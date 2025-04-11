import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import base64

# Add background image using local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to set background
add_bg_from_local("background.jpg")

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('Crop_recommendation.csv')
    return df

# Train and return model and scaler
@st.cache_resource
def train_model(data):
    X = data.drop('label', axis=1)
    y = data['label']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# Load data and model
data = load_data()
model, scaler = train_model(data)

# App layout
st.title("Crop Recommendation System")
st.markdown("### Enter the soil and weather conditions to get the best crop to grow.")

# Input fields
N = st.number_input('Nitrogen (N)', min_value=0.0, value=50.0)
P = st.number_input('Phosphorus (P)', min_value=0.0, value=50.0)
K = st.number_input('Potassium (K)', min_value=0.0, value=50.0)
temperature = st.number_input('Temperature (°C)', value=25.0)
humidity = st.number_input('Humidity (%)', value=60.0)
ph = st.number_input('pH Level', value=6.5)
rainfall = st.number_input('Rainfall (mm)', value=100.0)

# Predict button
if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.markdown(
    f"<h4 style='color: yellow;'> Recommended Crop: <b>{prediction[0].capitalize()}</b></h4>",
    unsafe_allow_html=True)