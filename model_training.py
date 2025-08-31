import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

# This script trains a simple CNN on CIFAR-10 dataset
# In a real scenario, you would replace this with your own dataset

def create_and_train_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert class vectors to binary class matrices
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # Define the model architecture
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Train the model
    model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=10,
        validation_data=(x_test, y_test),
        shuffle=True
    )
    
    # Save the model
    model.save('models/image_classifier.h5')
    print("Model trained and saved successfully!")
    
    return model

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    create_and_train_model()
