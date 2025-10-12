import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
from PIL import Image
import io

# Set page title and favicon
st.set_page_config(
    page_title="CAPTCHA Predictor",
    page_icon="🔍",
    layout="centered"
)

# Constants
IMG_HEIGHT = 50
IMG_WIDTH = 200
CAPTCHA_LENGTH = 5
CHARACTERS = '0123456789abcdefghijklmnopqrstuvwxyz'

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3 and image.shape[2] >= 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to the expected dimensions
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize pixel values
    image = image / 255.0
    
    # Reshape for the model - only add one dimension for batch
    image = np.expand_dims(image, axis=0)
    
    # Add channel dimension if not present
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=-1)
    
    return image

# Function to decode the prediction
def decode_prediction(prediction):
    char_indices = {}
    for i, char in enumerate(CHARACTERS):
        char_indices[i] = char
    
    captcha_text = ""
    for i in range(CAPTCHA_LENGTH):
        index = np.argmax(prediction[i])
        captcha_text += char_indices[index]
    
    return captcha_text

# Main function
def main():
    st.title("CAPTCHA Recognition App")
    st.write("Upload a CAPTCHA image and the model will predict the text")
    
    # Model loading with caching to avoid reloading on every rerun
    @st.cache_resource
    def load_captcha_model():
        model_path = 'captcha_model.h5'
        if os.path.exists(model_path):
            return load_model(model_path)
        else:
            st.error("Model file not found. Please make sure 'captcha_model.h5' exists in the app directory.")
            return None
    
    model = load_captcha_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CAPTCHA image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded CAPTCHA Image', use_column_width=True)
        
        # Convert PIL Image to numpy array
        image_array = np.array(image)
        
        # Add a predict button
        if st.button('Predict CAPTCHA'):
            with st.spinner('Processing...'):
                # Preprocess the image
                processed_image = preprocess_image(image_array)
                
                # Make prediction
                if model is not None:
                    predictions = model.predict(processed_image)
                    
                    # Decode the prediction
                    predicted_text = decode_prediction(predictions)
                    
                    # Display result
                    st.success(f"Predicted CAPTCHA text: {predicted_text}")
                    
                    # Show confidence levels
                    st.subheader("Confidence Levels")
                    for i in range(CAPTCHA_LENGTH):
                        char_confidence = np.max(predictions[i]) * 100
                        st.progress(int(char_confidence))
                        st.write(f"Character {i+1}: {char_confidence:.2f}%")
    
    # Add instructions
    with st.expander("How to use"):
        st.write("""
        1. Upload a CAPTCHA image using the file uploader above
        2. Click the 'Predict CAPTCHA' button
        3. The model will process the image and display the predicted text
        4. The confidence level for each character will be shown below the prediction
        """)
    
    # Add information about the model
    with st.expander("About the Model"):
        st.write("""
        This app uses a Convolutional Neural Network (CNN) trained on CAPTCHA images.
        The model achieves the following accuracy:
        - Character 1: ~98.6%
        - Character 2: ~87.9%
        - Character 3: ~82.2%
        - Character 4: ~80.8%
        - Character 5: ~83.2%
        """)

if __name__ == "__main__":
    main()
