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

# Function to get online image of the predicted crop
def get_online_image(crop_name):
    crop_images = {
        'rice': 'https://www.pexels.com/photo/close-up-photo-of-white-rice-grains-4110251/',  # Replace with a 4K image URL
        'maize': 'https://www.pexels.com/photo/person-holding-corn-from-its-tree-3307282/',  # Replace with a 4K image URL
        'chickpea': 'https://images.unsplash.com/photo-1596188771903-b16e49d5b4e1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80',  # Replace with a 4K image URL
        'kidneybeans': 'https://images.unsplash.com/photo-1582227257150-7a2202d0d081?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80',  # Replace with a 4K image URL
        'pigeonpeas': 'https://images.unsplash.com/photo-1569159585160-381fdf6fce9b?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80',  # Replace with a 4K image URL
        # Add more crops and URLs as needed
    }
    return crop_images.get(crop_name.lower(), 'https://via.placeholder.com/150')  # Placeholder for unknown crops

# Streamlit Web App
def main():
    st.set_page_config(page_title="Smart Crop and Fertilizer Recommendations", layout="wide")

    # Display the initial background image before prediction
    st.image("image.png", use_column_width=True)  # Display the uploaded image as the background before prediction

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
            
            # Display the online crop image after prediction
            st.image(get_online_image(crop), caption=f"Predicted Crop: {crop.capitalize()}", use_column_width=True)

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

            # Visualization: Bar chart of nutrient levels
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Nutrient Levels")
            fig, ax = plt.subplots()
            ax.bar(['Nitrogen', 'Phosphorus', 'Potassium'], [nitrogen, phosphorus, potassium], color=['#FF6347', '#FFD700', '#32CD32'])
            ax.set_ylabel('Level')
            ax.set_title('Nutrient Levels for Recommended Crop')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Visualization: Scatter plot for environmental factors
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Environmental Factors vs Recommended Crop")
            fig3, ax3 = plt.subplots()
            sns.scatterplot(x=df['temperature'], y=df['humidity'], hue=df['label'], ax=ax3, palette='coolwarm')
            ax3.scatter(temperature, humidity, color='red', s=100)  # Highlight the input values with a larger red dot
            ax3.set_xlabel('Temperature (°C)')
            ax3.set_ylabel('Humidity (%)')
            ax3.set_title('Temperature vs Humidity')
            st.pyplot(fig3)
            st.markdown('</div>', unsafe_allow_html=True)

# Run the web app
if __name__ == '__main__':
    main()

    
