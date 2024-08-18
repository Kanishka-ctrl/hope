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

# Function to predict crop
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction[0]

# Function to load and display an image of the predicted crop
def show_crop_image(crop_name):
    image_path = os.path.join('crop_images', crop_name.lower()+'.jpg')
    if os.path.exists(image_path):
        st.image(image_path, caption=f"Recommended crop: {crop_name.capitalize()}", use_column_width=True)
    else:
        st.warning("Image not found for the predicted crop.")

# Function to get crop rotation tips
def get_crop_rotation_tips(crop):
    return crop_rotation_tips.get(crop, crop_rotation_tips['default'])

# Streamlit Web App
def main():
    st.set_page_config(page_title="Smart Crop and Fertilizer Recommendations", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #f5f5f5;
            color: #333;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        .sidebar .sidebar-content h1, h2, h3, h4, h5, h6 {
            color: #2c6e49;
        }
        .st-header {
            font-family: 'Arial', sans-serif;
            color: #2c6e49;
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 20px;
        }
        .st-subheader {
            color: #4CAF50;
            font-size: 1.5rem;
        }
        .stNumberInput>div>div>input {
            background-color: #fff;
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
        </style>
    """, unsafe_allow_html=True)

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
            
            # Display crop image
            st.image('/mnt/data/image.png', use_column_width=True)
            
            # Display crop and fertilizer recommendations
            st.header(f"Recommended Crop: {crop.capitalize()}")
            st.subheader("Fertilizer Recommendation")
            fertilizer = fertilizer_recommendation(crop, nitrogen, phosphorus, potassium)
            st.write(fertilizer)
            
            # Display Crop Rotation Tips
            st.subheader("Crop Rotation Tips")
            rotation_tips = get_crop_rotation_tips(crop)
            st.write(rotation_tips)
            
            # Show crop image if available
            show_crop_image(crop)
            
            # Visualization: Bar chart of nutrient levels
            st.subheader("Nutrient Levels")
            fig, ax = plt.subplots()
            ax.bar(['Nitrogen', 'Phosphorus', 'Potassium'], [nitrogen, phosphorus, potassium], color=['#FF6347', '#FFD700', '#32CD32'])
            ax.set_ylabel('Level')
            ax.set_title('Nutrient Levels for Recommended Crop')
            st.pyplot(fig)
            
            # Visualization: Pie chart of crop distribution in the dataset
            st.subheader("Crop Distribution in Dataset")
            crop_counts = df['label'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(crop_counts, labels=crop_counts.index
