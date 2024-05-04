from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Preprocess the input image
def preprocess_image(image):
    resized_image = cv2.resize(image, (64, 64))
    input_img = np.expand_dims(resized_image, axis=0)
    return input_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']

        if file:
            # Read the image file
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Preprocess the image
            input_img = preprocess_image(img)

            # Make prediction
            result = model.predict(input_img)
            prediction = np.argmax(result)

            # Render result page
            return render_template('result.html', prediction=prediction)
        else:
            return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True)
