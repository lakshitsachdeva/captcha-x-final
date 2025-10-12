import numpy as np
import cv2
import os
import string
from tensorflow.keras.models import load_model

# Constants
IMG_HEIGHT = 50
IMG_WIDTH = 200
CAPTCHA_LENGTH = 5
CHARACTERS = string.digits + string.ascii_lowercase

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

def decode_prediction(pred):
    """Convert model prediction to text"""
    char_indices = {i: char for i, char in enumerate(CHARACTERS)}
    captcha_text = ""
    
    for i in range(CAPTCHA_LENGTH):
        char_idx = np.argmax(pred[i][0])
        captcha_text += char_indices[char_idx]
    
    return captcha_text

def predict_captcha(model_path, image_path):
    """Predict the text in a captcha image"""
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_img)
    
    # Decode prediction
    captcha_text = decode_prediction(predictions)
    
    return captcha_text

def main():
    # Path to the saved model
    model_path = 'captcha_model.h5'
    
    # Path to the samples directory
    samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captcha_dataset', 'samples')
    
    # Get a few sample images for testing
    image_files = [f for f in os.listdir(samples_path) if f.endswith('.png')][:5]
    
    for image_file in image_files:
        image_path = os.path.join(samples_path, image_file)
        true_label = os.path.splitext(image_file)[0]
        
        try:
            predicted_text = predict_captcha(model_path, image_path)
            print(f"Image: {image_file}, True: {true_label}, Predicted: {predicted_text}")
        except Exception as e:
            print(f"Error predicting captcha{image_file}: {e}")

if __name__ == "__main__":
    main()