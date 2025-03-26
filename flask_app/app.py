import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
import cv2
from PIL import Image
import io
import tensorflow as tf
import uuid

app = Flask(__name__)

# Load the SavedModel
model = tf.saved_model.load(r'C:\Sarvesh\Capstone\flask_app\model\model_eye_diseases7_denoised')
predict_function = model.signatures["serving_default"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Read and preprocess the image
            img = Image.open(file)
            img = np.array(img)
            img_resized = cv2.resize(img, (320, 320))  # Resize to match model input size
            img_input = np.expand_dims(img_resized, axis=0)

            # Make prediction using the loaded model
            input_tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)
            prediction = predict_function(input_tensor)

            # Access the correct key for the output
            predicted_mask = prediction['conv2d_37'].numpy()[0, :, :, 0]  # Adjust this based on the actual key

            # Merge prediction over the original OCT image
            merged_img = img_resized.copy()

            # Simple thresholding to visualize the predicted regions
            merged_img[predicted_mask > 0.5] = [0, 0, 255]  # Red for predicted regions
            
            # Save the resulting image for Flask to serve
            unique_filename = f"{uuid.uuid4().hex}.png"
            output_path = os.path.join('static', unique_filename)
            cv2.imwrite(output_path, merged_img)

            # Return the path of the saved image for displaying on the webpage
            return render_template('index.html', image_url=f"/static/{unique_filename}")

if __name__ == '__main__':
    app.run(debug=True)
