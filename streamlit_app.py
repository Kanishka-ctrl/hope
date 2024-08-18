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

# Custom Fertilizer Recommendations
def fertilizer_recommendation(crop, nitrogen, phosphorus, potassium):
    if crop == 'rice':
        if nitrogen < 50:
            return "Urea, DAP - Increase Urea due to low nitrogen levels."
        else:
            return "Urea, DAP - Standard application."
    elif crop == 'maize':
        if phosphorus < 30:
            return "SSP, MOP - Increase SSP due to low phosphorus."
        else:
            return "Urea, SSP, MOP - Standard application."
    elif crop == 'chickpea':
        return "DAP, Potash, Zinc Sulphate - Apply Potash to improve crop yield."
    elif crop == 'kidneybeans':
        return "Urea, SSP, MOP - Apply MOP at the flowering stage."
    elif crop == 'pigeonpeas':
        return "SSP, Urea, MOP - Balanced application during growth."
    else:
        return "Standard Fertilizer Recommendation for your crop."

# Crop Rotation Tips
crop_rotation_tips = {
    'rice': "After rice, consider planting legumes like mungbean or lentil to improve soil nitrogen levels.",
    'maize': "Rotate maize with legumes such as soybeans or groundnut to help restore soil fertility.",
    'chickpea': "Follow chickpea with cereal crops like wheat or barley to balance nutrient usage.",
    'kidneybeans': "Rotate kidney beans with cereal crops or root vegetables to prevent soil depletion.",
    'pigeonpeas': "Pigeon peas can be rotated with cotton or millet to break the pest and disease cycle.",
    'default': "Consider rotating with a legume or cereal crop to maintain soil health."
}

# Function to get crop rotation tips
def get_crop_rotation_tips(crop):
    return crop_rotation_tips.get(crop.lower(), crop_rotation_tips['default'])

# Function to predict crop
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction[0]

# Streamlit Web App
def main():
    st.set_page_config(page_title="Smart Crop and Fertilizer Recommendations", layout="wide")

    # Set a general background image
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("crop.jpg"); /* Use the local image file */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .sidebar .sidebar-content {
        background-color: rgba(0, 0, 0, 0.6); /* Add transparency */
        padding: 20px;
        border-radius: 10px;
    }
    .st-header, .st-subheader {
        text-shadow: 2px 2px 5px #000;
    }
    .stNumberInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.7);
        color: #333;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-size: 1rem;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Sidebar for inputs
    st.sidebar.header("Enter Crop Details")
    nitrogen = st.sidebar.number_input("Nitrogen", min_value=0.0, max_value=140.0, value=0.0, step=0.1)
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
            
            # Display crop image using the local 'crop.jpg' file
            st.image("crop.jpg", caption=f"Predicted Crop: {crop.capitalize()}", use_column_width=True)

            # Display crop and fertilizer recommendations in a card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header(f"Recommended Crop: {crop.capitalize()}")
            st.subheader("Fertilizer Recommendation")
            fertilizer = fertilizer_recommendation(crop, nitrogen, phosphorus, potassium)
            st.write(fertilizer)
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
