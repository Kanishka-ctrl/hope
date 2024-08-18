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
        'image': 'https://www.pexels.com/photo/close-up-photo-of-white-rice-grains-4110251/'
    },
    'maize': {
        'season': 'Kharif (June to October), Rabi (November to February)',
        'regions': 'Karnataka, Maharashtra, Madhya Pradesh, Uttar Pradesh',
        'soil': 'Well-drained fertile soil, Sandy loam soil',
        'image': 'https://www.pexels.com/photo/person-holding-corn-from-its-tree-3307282/'
    },
    'chickpea': {
        'season': 'Rabi (October to March)',
        'regions': 'Madhya Pradesh, Maharashtra, Rajasthan, Uttar Pradesh',
        'soil': 'Well-drained loamy soil, Sandy loam soil',
        'image': 'https://harvesttotable.com/wp-content/uploads/2009/04/Chickpeas-bigstock-Fresh-Green-Chickpeas-Field-337227136-scaled-1024x683.jpg'
    },
    'kidneybeans': {
        'season': 'Kharif (June to October)',
        'regions': 'Uttarakhand, Himachal Pradesh, Jammu & Kashmir',
        'soil': 'Loamy soil, Sandy loam soil',
        'image': 'https://www.pexels.com/photo/wooden-bowl-of-brown-kidney-beans-on-a-table-18233225/'
    },
    'pigeonpeas': {
        'season': 'Kharif (June to October)',
        'regions': 'Maharashtra, Uttar Pradesh, Madhya Pradesh, Karnataka',
        'soil': 'Well-drained loamy soil',
        'image': 'https://th.bing.com/th/id/OIP.LrzXYTsEDQyICA1XeTJbrAHaEK?rs=1&pid=ImgDetMain'
    },
    'mothbeans': {
        'season': 'Kharif (June to September)',
        'regions': 'Rajasthan, Maharashtra, Gujarat, Haryana',
        'soil': 'Sandy loam soil',
        'image': 'https://media.istockphoto.com/id/466255398/photo/moth-beans-background.jpg?s=612x612&w=0&k=20&c=xLp23CNioXUSRh_HDlGj6Z0-zssGlD8UqysckgWgAKM='  # Replace with an actual image URL
    },
    'mungbean': {
        'season': 'Kharif (June to October), Rabi (March to June)',
        'regions': 'Maharashtra, Andhra Pradesh, Rajasthan, Karnataka',
        'soil': 'Loamy soil, Sandy loam soil',
        'image': 'https://149366112.v2.pressablecdn.com/wp-content/uploads/2018/10/mung-beans-390724_1920.jpg'  # Replace with an actual image URL
    },
    'blackgram': {
        'season': 'Kharif (June to October), Rabi (November to March)',
        'regions': 'Maharashtra, Andhra Pradesh, Uttar Pradesh, Madhya Pradesh',
        'soil': 'Loamy soil, Sandy soil',
        'image': 'https://www.bing.com/ck/a?!&&p=28acd41b2bac0ccb1badea0404b9be39c285fa59ba82cc0b53a9fec177565d6eJmltdHM9MTcyMzkzOTIwMCZpZ3VpZD0yNWE5OGFhYy1lMjdmLTY0N2EtMTFkMy05ZTEzZTM3OTY1YjYmaW5zaWQ9NTU4Ng&ptn=3&ver=2&hsh=4&fclid=25a98aac-e27f-647a-11d3-9e13e37965b6&u=a1L2ltYWdlcy9zZWFyY2g_cT1ibGFja2dyYW0lMjBpbWFnZSUyMDRrJkZPUk09SVFGUkJBJmlkPUFFQjc2OTkzNDI1REI5Q0FCREI5MkJBNzQ5RENCQjg3NzNCMkQ0ODg&ntb=1'  # Replace with an actual image URL
    },
    'lentil': {
        'season': 'Rabi (November to March)',
        'regions': 'Madhya Pradesh, Uttar Pradesh, Bihar, Rajasthan',
        'soil': 'Loamy soil, Clayey soil',
        'image': 'https://www.bing.com/images/search?view=detailV2&ccid=u9NnikAq&id=B74642D34740661ED97E4027026AB4CB74F1B793&thid=OIP.u9NnikAqO7IBOPrV3FME_QHaE8&mediaurl=https%3a%2f%2fc1.wallpaperflare.com%2fpreview%2f879%2f102%2f375%2fchana-daal-lentils-indian.jpg&exph=607&expw=910&q=lentil+image+4k&simid=608004869829381599&FORM=IRPRST&ck=571319BDC9246540EE37826554051910&selectedIndex=8&itb=0'  # Replace with an actual image URL
    },
    'pomegranate': {
        'season': 'June to September',
        'regions': 'Maharashtra, Gujarat, Rajasthan, Karnataka',
        'soil': 'Well-drained loamy soil',
        'image': 'https://www.pexels.com/photo/red-round-fruit-65256/'  # Replace with an actual image URL
    },
    'banana': {
        'season': 'Throughout the year',
        'regions': 'Tamil Nadu, Maharashtra, Andhra Pradesh, Gujarat',
        'soil': 'Loamy soil, Alluvial soil',
        'image': 'https://www.pexels.com/photo/banana-tree-802783/'  # Replace with an actual image URL
    },
    'mango': {
        'season': 'Summer (March to June)',
        'regions': 'Uttar Pradesh, Andhra Pradesh, Maharashtra, Karnataka',
        'soil': 'Well-drained loamy soil, Alluvial soil',
        'image': 'https://th.bing.com/th/id/OIP.Q9RjS1V0q0xFtQMFF_kBkQHaE7?rs=1&pid=ImgDetMain'  # Replace with an actual image URL
    },
    'grapes': {
        'season': 'Winter (December to February)',
        'regions': 'Maharashtra, Karnataka, Tamil Nadu, Andhra Pradesh',
        'soil': 'Sandy loam soil, Clayey soil',
        'image': 'https://www.pexels.com/photo/green-grapes-60021/'  # Replace with an actual image URL
    },
    'watermelon': {
        'season': 'Summer (March to June)',
        'regions': 'Uttar Pradesh, Punjab, Haryana, Madhya Pradesh',
        'soil': 'Sandy loam soil, Alluvial soil',
        'image': 'https://th.bing.com/th/id/OIP.B5cF-j2I3zwSFxtIhHfLLgHaFj?rs=1&pid=ImgDetMain'  # Replace with an actual image URL
    },
    'muskmelon': {
        'season': 'Summer (March to June)',
        'regions': 'Uttar Pradesh, Punjab, Haryana, Rajasthan',
        'soil': 'Sandy loam soil, Alluvial soil',
        'image': 'https://wallpapercave.com/wp/wp9785653.jpg'  # Replace with an actual image URL
    },
    'apple': {
        'season': 'Autumn (September to November)',
        'regions': 'Jammu & Kashmir, Himachal Pradesh, Uttarakhand',
        'soil': 'Well-drained loamy soil',
        'image': 'https://www.pexels.com/photo/red-apples-635705/'  # Replace with an actual image URL
    },
    'orange': {
        'season': 'Winter (December to February)',
        'regions': 'Maharashtra, Andhra Pradesh, Punjab, Karnataka',
        'soil': 'Loamy soil, Alluvial soil',
        'image': 'https://www.pexels.com/photo/close-up-of-fruits-in-bowl-327098/'  # Replace with an actual image URL
    },
    'papaya': {
        'season': 'Throughout the year',
        'regions': 'Maharashtra, Gujarat, Tamil Nadu, West Bengal',
        'soil': 'Loamy soil, Alluvial soil',
        'image': 'https://images7.alphacoders.com/376/thumb-1920-376682.jpg'  # Replace with an actual image URL
    },
    'coconut': {
        'season': 'Throughout the year',
        'regions': 'Kerala, Tamil Nadu, Karnataka, Andhra Pradesh',
        'soil': 'Sandy loam soil, Alluvial soil',
        'image': 'https://www.pexels.com/photo/selective-focus-photo-of-coconuts-1424457/'  # Replace with an actual image URL
    },
    'cotton': {
        'season': 'Kharif (June to September)',
        'regions': 'Maharashtra, Gujarat, Andhra Pradesh, Punjab',
        'soil': 'Black soil, Sandy loam soil',
        'image': 'https://wallpapercave.com/wp/wp2131890.jpg'  # Replace with an actual image URL
    },
    'jute': {
        'season': 'Kharif (June to September)',
        'regions': 'West Bengal, Bihar, Assam, Odisha',
        'soil': 'Alluvial soil',
        'image': 'https://www.pexels.com/photo/close-up-of-a-texture-and-pattern-of-a-woven-jute-7717489/'  # Replace with an actual image URL
    },
     'coffee': {
        'season': 'Winter (November to March)',
        'regions': 'Karnataka, Kerala, Tamil Nadu',
        'soil': 'Well-drained loamy soil, Sandy loam soil',
        'image': 'https://th.bing.com/th/id/OIP.TIK3xwcvcfqfDRGSE5kjygHaE7?rs=1&pid=ImgDetMain'  # Replace with an actual image URL
     }
}

# Function to predict crop
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction = RF_Model_pkl.predict(np.array([nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]).reshape(1, -1))
    return prediction[0]

# Function to get online image and additional info of the predicted crop
def get_crop_info(crop_name):
    return crop_info.get(crop_name.lower(), {
        'season': 'Unknown',
        'regions': 'Unknown',
        'soil': 'Unknown',
        'image': 'https://via.placeholder.com/150'  # Placeholder image
    })

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
            crop_details = get_crop_info(crop)

            # Display the online crop image after prediction
            st.image(crop_details['image'], caption=f"Predicted Crop: {crop.capitalize()}", use_column_width=True)

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
