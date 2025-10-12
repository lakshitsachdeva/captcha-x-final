import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import string
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_HEIGHT = 50
IMG_WIDTH = 200
BATCH_SIZE = 32
EPOCHS = 60  
CAPTCHA_LENGTH = 5
CHARACTERS = string.digits + string.ascii_lowercase

# Paths
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'captcha_dataset')
SAMPLES_PATH = os.path.join(DATASET_PATH, 'samples')
TRAIN_CSV = os.path.join(DATASET_PATH, 'ml_data_3188.csv')
TEST_CSV = os.path.join(DATASET_PATH, 'ml_test_3188.csv')

def load_data():
    """Load and preprocess the captcha dataset"""
    print("Loading dataset...")
    
    # Get all image filenames
    image_files = [f for f in os.listdir(SAMPLES_PATH) if f.endswith('.png') or f.endswith('.jpg')]
    
    # Extract labels from filenames (assuming filenames like 'abcd1.png')
    labels = [os.path.splitext(f)[0] for f in image_files]
    
    # Filter out labels that don't have exactly CAPTCHA_LENGTH characters
    valid_indices = [i for i, label in enumerate(labels) if len(label) == CAPTCHA_LENGTH]
    valid_image_files = [image_files[i] for i in valid_indices]
    valid_labels = [labels[i] for i in valid_indices]
    
    print(f"Found {len(valid_image_files)} valid images with {CAPTCHA_LENGTH} characters")
    
    # Load and preprocess images
    images = []
    final_labels = []
    
    for img_file, label in zip(valid_image_files, valid_labels):
        img_path = os.path.join(SAMPLES_PATH, img_file)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                img = img / 255.0 
                images.append(img)
                final_labels.append(label)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    # Convert to numpy arrays
    X = np.array(images)
    X = X.reshape(X.shape[0], IMG_HEIGHT, IMG_WIDTH, 1)  # Add channel dimension
    
    # One-hot encode the labels
    char_indices = {char: i for i, char in enumerate(CHARACTERS)}
    
    # Initialize the encoded labels array with zeros
    encoded_labels = np.zeros((len(final_labels), CAPTCHA_LENGTH, len(CHARACTERS)))
    
    # Fill in the one-hot encoded values
    for i, label in enumerate(final_labels):
        for j, char in enumerate(label):
            if char in char_indices:
                encoded_labels[i, j, char_indices[char]] = 1.0
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def build_model():
    """Build the CNN model for captcha recognition"""
    # Input layer
    input_img = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    
    # CNN layers
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten layer
    x = layers.Flatten()(x)
    
    # Create separate branches for each character
    outputs = []
    for i in range(CAPTCHA_LENGTH):
        # Dense layer for each character
        dense = layers.Dense(64)(x)
        dropout = layers.Dropout(0.2)(dense)
        out = layers.Dense(len(CHARACTERS), activation='softmax', name=f'char_{i}')(dropout)
        outputs.append(out)
    
    # Create and compile model
    model = Model(inputs=input_img, outputs=outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model

def train_model(model, X_train, X_test, y_train, y_test):
    """Train the model with data augmentation and save it"""
    # Prepare callbacks
    checkpoint = callbacks.ModelCheckpoint(
        filepath='captcha_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001
    )
    
    # Prepare the target data
    y_train_list = [y_train[:, i] for i in range(CAPTCHA_LENGTH)]
    y_test_list = [y_test[:, i] for i in range(CAPTCHA_LENGTH)]
    
    # Create data generator for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Train model with data augmentation
    # We'll use fit() instead of flow() to avoid shape mismatch issues
    history = model.fit(
        X_train, y_train_list,
        validation_data=(X_test, y_test_list),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    return model
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    for i in range(CAPTCHA_LENGTH):
        plt.plot(history.history[f'char_{i}_accuracy'], label=f'Char {i+1} Accuracy')
    plt.title('Character Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance"""
    y_test_list = [y_test[:, i] for i in range(CAPTCHA_LENGTH)]
    scores = model.evaluate(X_test, y_test_list, verbose=1)
    
    print("Test Loss:", scores[0])
    for i in range(CAPTCHA_LENGTH):
        print(f"Character {i+1} Accuracy: {scores[i+CAPTCHA_LENGTH+1]}")
    
    # Make predictions on a few test samples
    predictions = model.predict(X_test[:5])
    
    # Convert predictions to characters
    index_to_char = {i: char for i, char in enumerate(CHARACTERS)}
    for i in range(5):
        predicted_captcha = ""
        for j in range(CAPTCHA_LENGTH):
            char_index = np.argmax(predictions[j][i])
            predicted_captcha += index_to_char[char_index]
        
        # Get true captcha
        true_captcha = ""
        for j in range(CAPTCHA_LENGTH):
            char_index = np.argmax(y_test[i, j])
            true_captcha += index_to_char[char_index]
        
        print(f"Sample {i+1}: True: {true_captcha}, Predicted: {predicted_captcha}")

def main():
    """Main function to run the training pipeline"""
    print("Starting captcha recognition model training...")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data()
    
    # Build model
    model = build_model()
    
    # Train model
    model = train_model(model, X_train, X_test, y_train, y_test)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    print("Training completed successfully. Model saved as 'captcha_model.h5'")

if __name__ == "__main__":
    main()
