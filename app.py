from flask import Flask, render_template, request, redirect, url_for, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
import cv2

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load the model (will be loaded once when the app starts)
model = None

def load_model_once():
    global model
    if model is None:
        try:
            model = load_model('models/image_classifier.h5')
            print("Model loaded successfully!")
        except:
            print("Could not load model. Please train the model first.")
            model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    # Resize image to 32x32 (the size our model expects)
    image = image.resize((32, 32))
    # Convert to array and normalize
    image = img_to_array(image)
    image = image.astype('float32') / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(request.url)
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Load model if not already loaded
        load_model_once()
        if model is None:
            flash('Model not available. Please train the model first.')
            return redirect(request.url)
        
        # Read and preprocess the image
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = round(100 * np.max(predictions[0]), 2)
        
        # Save the uploaded file
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        image.save(file_path)
        
        # Prepare results
        result = {
            'class': class_names[predicted_class],
            'confidence': confidence,
            'filename': file.filename
        }
        
        return render_template('result.html', result=result)
    
    else:
        flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
        return redirect(request.url)

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Load model at startup
    load_model_once()
    
    # Run the app
    app.run(debug=True)
