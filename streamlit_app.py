import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Extract features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain, Ytrain)

# Save the trained model
RF_pkl_filename = 'RF.pkl'
with open(RF_pkl_filename, 'wb') as file:
    pickle.dump(RF, file)

# Load the model
RF_Model_pkl = pickle.load(open('RF.pkl', 'rb'))

# Crop Information: Seasons, Regions, Soil Types
crop_info = {
    'rice': {
        'season': 'Kharif (June to November)',
        'regions': 'West Bengal, Punjab, Uttar Pradesh, Bihar',
        'soil': 'Clayey soil, Loamy soil',
    },
    'maize': {
        'season': 'Kharif (June to October), Rabi (November to February)',
        'regions': 'Karnataka, Maharashtra, Madhya Pradesh, Uttar Pradesh',
        'soil': 'Well-drained fertile soil, Sandy loam soil',
    },
    'chickpea': {
        'season': 'Rabi (October to March)',
        'regions': 'Madhya Pradesh, Maharashtra, Rajasthan, Uttar Pradesh',
        'soil': 'Well-drained loamy soil, Sandy loam soil',
    },
    'kidneybeans': {
        'season': 'Kharif (June to October)',
        'regions': 'Uttarakhand, Himachal Pradesh, Jammu & Kashmir',
        'soil': 'Loamy soil, Sandy loam soil',
    },
    'pigeonpeas': {
        'season': 'Kharif (June to October)',
        'regions': 'Maharashtra, Uttar Pradesh, Madhya Pradesh, Karnataka',
        'soil': 'Well-drained loamy soil',
    },
    'mothbeans': {
        'season': 'Kharif (June to September)',
        'regions': 'Rajasthan, Maharashtra, Gujarat, Haryana',
        'soil': 'Sandy loam soil',
    },
    'mungbean': {
        'season': 'Kharif (June to October), Rabi (March to June)',
        'regions': 'Maharashtra, Andhra Pradesh, Rajasthan, Karnataka',
        'soil': 'Loamy soil, Sandy loam soil',
    },
    'blackgram': {
        'season': 'Kharif (June to October), Rabi (November to March)',
        'regions': 'Maharashtra, Andhra Pradesh, Uttar Pradesh, Madhya Pradesh',
        'soil': 'Loamy soil, Sandy soil',
    },
    'lentil': {
        'season': 'Rabi (November to March)',
        'regions': 'Madhya Pradesh, Uttar Pradesh, Bihar, Rajasthan',
        'soil': 'Loamy soil, Clayey soil',
    },
    'pomegranate': {
        'season': 'June to September',
        'regions': 'Maharashtra, Gujarat, Rajasthan, Karnataka',
        'soil': 'Well-drained loamy soil',
    },
    'banana': {
        'season': 'Throughout the year',
        'regions': 'Tamil Nadu, Maharashtra, Andhra Pradesh, Gujarat',
        'soil': 'Loamy soil, Alluvial soil',
    },
    'mango': {
        'season': 'Summer (March to June)',
        'regions': 'Uttar Pradesh, Andhra Pradesh, Maharashtra, Karnataka',
        'soil': 'Well-drained loamy soil, Alluvial soil',
    },
    'grapes': {
        'season': 'Winter (December to February)',
        'regions': 'Maharashtra, Karnataka, Tamil Nadu, Andhra Pradesh',
        'soil': 'Sandy loam soil, Clayey soil',
    },
    'watermelon': {
        'season': 'Summer (March to June)',
        'regions': 'Uttar Pradesh, Punjab, Haryana, Madhya Pradesh',
        'soil': 'Sandy loam soil, Alluvial soil',
    },
    'muskmelon': {
        'season': 'Summer (March to June)',
        'regions': 'Uttar Pradesh, Punjab, Haryana, Rajasthan',
        'soil': 'Sandy loam soil, Alluvial soil',
    },
    'apple': {
        'season': 'Autumn (September to November)',
        'regions': 'Jammu & Kashmir, Himachal Pradesh, Uttarakhand',
        'soil': 'Well-drained loamy soil',
    },
    'orange': {
        'season': 'Winter (December to February)',
        'regions': 'Maharashtra, Andhra Pradesh, Punjab, Karnataka',
        'soil': 'Loamy soil, Alluvial soil',
    },
    'papaya': {
        'season': 'Throughout the year',
        'regions': 'Maharashtra, Gujarat, Tamil Nadu, West Bengal',
        'soil': 'Loamy soil, Alluvial soil',
    },
    'coconut': {
        'season': 'Throughout the year',
        'regions': 'Kerala, Tamil Nadu, Karnataka, Andhra Pradesh',
        'soil': 'Sandy loam soil, Alluvial soil',
    },
    'cotton': {
        'season': 'Kharif (June to September)',
        'regions': 'Maharashtra, Gujarat, Andhra Pradesh, Punjab',
        'soil': 'Black soil, Sandy loam soil',
    },
    'jute': {
        'season': 'Kharif (June to September)',
        'regions': 'West Bengal, Bihar, Assam, Odisha',
        'soil': 'Alluvial soil',
    },
    'coffee': {
        'season': 'Winter (November to March)',
        'regions': 'Karnataka, Kerala, Tamil Nadu',
        'soil': 'Well-drained loamy soil, Sandy loam soil',
    }
}

# Function to predict crop
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction[0]

# Function to get crop info
def get_crop_info(crop_name):
    return crop_info.get(crop_name.lower(), {
        'season': 'Unknown',
        'regions': 'Unknown',
        'soil': 'Unknown'
    })

# Function to provide fertilizer recommendation
def fertilizer_recommendation(crop, nitrogen, phosphorus, potassium):
    if crop == 'rice':
        return "Urea, DAP - Standard application."
    elif crop == 'maize':
        return "Urea, SSP, MOP - Standard application."
    elif crop == 'chickpea':
        return "DAP, Potash, Zinc Sulphate - Apply Potash to improve crop yield."
    elif crop == 'kidneybeans':
        return "Urea, SSP, MOP - Apply MOP at the flowering stage."
    elif crop == 'pigeonpeas':
        return "SSP, Urea, MOP - Balanced application during growth."
    else:
        return "Standard Fertilizer Recommendation for your crop."

# Function to provide crop rotation tips
def get_crop_rotation_tips(crop):
    rotation_tips = {
        'rice': "After rice, consider planting legumes like mungbean or lentil to improve soil nitrogen levels.",
        'maize': "Rotate maize with legumes such as soybeans or groundnut to help restore soil fertility.",
        'chickpea': "Follow chickpea with cereal crops like wheat or barley to balance nutrient usage.",
        'kidneybeans': "Rotate kidney beans with cereal crops or root vegetables to prevent soil depletion.",
        'pigeonpeas': "Pigeon peas can be rotated with cotton or millet to break the pest and disease cycle.",
        'default': "Consider rotating with a legume or cereal crop to maintain soil health."
    }
    return rotation_tips.get(crop.lower(), rotation_tips['default'])

# Streamlit Web App
def main():
    st.set_page_config(page_title="Smart Crop and Fertilizer Recommendations", layout="wide")

    # Display the initial background image before prediction
    try:
        st.image("image.png", use_column_width=True)  # Attempt to display the uploaded image as the background
    except Exception as e:
        st.error("Error loading background image.")
        st.write(str(e))

    # Sidebar for inputs
    st.sidebar.header("Enter Crop Details")
    nitrogen = st.sidebar.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
    phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.
    phosphorus = st.sidebar.number_input("Phosphorus", min_value=0.0, max_value=145.0, value=0.0, step=0.1)
    potassium = st.sidebar.number_input("Potassium", min_value=0.0, max_value=205.0, value=0.0, step=0.1)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=51.0, value=0.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
    ph = st.sidebar.number_input("pH Level", min_value=0.0, max_value=14.0, value=0.0, step=0.1)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)

    # Predict and display results
    if st.sidebar.button("Predict"):
        if not np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).any() or np.isnan(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall])).any():
            st.error("Please fill in all input fields with valid values before predicting.")
        else:
            crop = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            crop_details = get_crop_info(crop)

            # Display the crop image after prediction
            try:
                st.image(crop_details['image'], caption=f"Predicted Crop: {crop.capitalize()}", use_column_width=True)
            except Exception as e:
                st.error("Error loading crop image.")
                st.write(str(e))

            # Display crop and fertilizer recommendations in a card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header(f"Recommended Crop: {crop.capitalize()}")
            st.subheader("Fertilizer Recommendation")
            fertilizer = fertilizer_recommendation(crop, nitrogen, phosphorus, potassium)
            st.write(fertilizer)
            st.subheader("Best Growing Season")
            st.write(crop_details['season'])
            st.subheader("Suitable Regions in India")
            st.write(crop_details['regions'])
            st.subheader("Preferred Soil Type")
            st.write(crop_details['soil'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display Crop Rotation Tips in a card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Crop Rotation Tips")
            rotation_tips = get_crop_rotation_tips(crop)
            st.write(rotation_tips)
            st.markdown('</div>', unsafe_allow_html=True)

# Run the web app
if __name__ == '__main__':
    main()
