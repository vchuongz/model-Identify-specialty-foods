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
import mysql.connector


def get_db_connection():
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'PBL4',
        'port': 3307
    }
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print(f"Lỗi kết nối cơ sở dữ liệu: {err}")
        return None


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model
model_path = 'train.h5'
model = load_model(model_path)

class_labels = [
    'Banh beo', 'Banh chung', 'Banh cuon',
    'Banh mi',
    'Banh trang nuong', 'Banh xeo',  'Bun dau mam tom',
     'Ca kho to',
    'Pho', 'Xoi xeo'
]


# Image preprocessing function
def preprocess_image_screenshot(img, target_size=(224, 224)):
    img = img.resize(target_size)  # Resize the image
    img_array = np.array(img)  # Convert the image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values to range [0, 1]
    return img_array



def preprocess_image_file(img, target_size=(224, 224)):
    img = img.resize(target_size)  # Resize the image
    img_array = np.array(img)  # Convert the image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values to range [0, 1]
    return img_array





def predict_image(model, img_array ,confidence_threshold=98):
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = class_labels[predicted_class]
    confidence = prediction[0][predicted_class] * 100
    if confidence < confidence_threshold:
        return "not found", confidence
    return predicted_label, confidence


@app.route('/predict', methods=['POST'])
def predict():
    # Decode the received image
    img_data = request.json['image']
    img_type = request.json.get('image_type', 'file')  # Nhận loại ảnh ('screenshot' hoặc 'file')

    img_data = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_data)).convert('RGB')

    # Chọn hàm tiền xử lý dựa trên loại ảnh
    if img_type == 'screenshot':
        img_array = preprocess_image_screenshot(img)
    else:
        img_array = preprocess_image_file(img)

    # Dự đoán
    predicted_label, confidence = predict_image(model, img_array)

    return jsonify({'label': predicted_label, 'confidence': confidence})


@app.route('/food-details', methods=['POST'])
def food_details():
    food_name = request.json['name']
    conn = get_db_connection()
    if conn is None:
        return jsonify({'error': 'Không thể kết nối tới cơ sở dữ liệu'}), 500

    cursor = conn.cursor(dictionary=True)

    try:
        query = "SELECT * FROM FOOD WHERE name = %s"
        cursor.execute(query, (food_name,))
        result = cursor.fetchone()
    except mysql.connector.Error as err:
        print(f"Lỗi khi thực thi truy vấn: {err}")
        return jsonify({'error': 'Truy vấn thất bại'}), 500
    finally:
        cursor.close()
        conn.close()

    if result:
        return jsonify(result)
    else:
        return jsonify({'error': 'Không tìm thấy thông tin món ăn'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



