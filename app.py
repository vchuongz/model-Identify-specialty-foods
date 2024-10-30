


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import cv2
import base64
import io
from PIL import Image, ImageEnhance
from flask_cors import CORS
from PIL import ImageEnhance, ImageFilter ,ImageEnhance


app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})


# Load the trained model
model_path = 'final_modeldk.h5'
model = load_model(model_path)

# Define class labels
class_labels = [
    'Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 'Banh cuon',
    'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi', 'Banh pia', 'Banh tet',
    'Banh trang nuong', 'Banh xeo', 'Bun bo Hue', 'Bun dau mam tom', 'Bun mam',
    'Bun rieu', 'Bun thit nuong', 'Ca kho to', 'Canh chua', 'Cau lau',
    'Chao long', 'Com tam', 'Goi cuon', 'Hu tieu', 'My quang', 'Nem chua',
    'Pho', 'Xoi xeo'
]

#Image preprocessing function
def preprocess_image(img, target_size=(150, 150)):
    # Resize the image
    img = img.resize(target_size)
    # Enhance image quality
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)  # Increase contrast


    enhancer_brightness = ImageEnhance.Brightness(img)
    img = enhancer_brightness.enhance(1.2)  # Slightly increase brightness

    for _ in range(2):  # Apply sharpening twice
        img = img.filter(ImageFilter.SHARPEN)

    # Apply sharpening filter
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array



# Prediction function
def predict_image(model, img_array):
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class] * 100
    return predicted_label, confidence





@app.route('/predict', methods=['POST'])
def predict():
    # Decode the received image
    img_data = request.json['image']
    img_data = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')

    # Preprocess and predict
    img_array = preprocess_image(img)
    predicted_label, confidence = predict_image(model, img_array)

    # Convert the image to OpenCV format for annotation
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.putText(img_cv, f"{predicted_label} ({confidence:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Encode the image to send back as base64
    _, buffer = cv2.imencode('.jpg', img_cv)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': img_base64, 'label': predicted_label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)




