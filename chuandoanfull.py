


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Tải mô hình đã huấn luyện
model_path = 'final_modeldk.h5'
model = load_model(model_path)

# Đường dẫn thư mục test
test_dir = r'D:\PBL4\Project\pbl4\datasets-BKĐN\Test'

# Danh sách nhãn lớp với 30 nhãn
class_labels = [
    'Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung', 'Banh cuon', 'Banh duc', 'Banh gio', 'Banh khot',
    'Banh mi',
    'Banh pia', 'Banh tet', 'Banh trang nuong', 'Banh xeo', 'Bun bo Hue', 'Bun dau mam tom', 'Bun mam', 'Bun rieu',
    'Bun thit nuong', 'Ca kho to',
    'Canh chua', 'Cau lau', 'Chao long', 'Com tam', 'Goi cuon', 'Hu tieu', 'My quang', 'Nem chua', 'Pho', 'Xoi xeo'
]


# Hàm tiền xử lý ảnh
def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)  # Thay đổi kích thước
    img_array = image.img_to_array(img)  # Chuyển ảnh thành mảng numpy
    img_array = np.expand_dims(img_array, axis=0)  # Thêm trục batch (batch_size = 1)
    img_array /= 255.0  # Chuẩn hóa giá trị pixel
    return img_array


# Hàm dự đoán nhãn
def predict_image(model, img_array, class_labels):
    prediction = model.predict(img_array)  # Dự đoán nhãn
    predicted_class = np.argmax(prediction, axis=1)  # Lấy chỉ số lớp dự đoán
    predicted_label = class_labels[predicted_class[0]]  # Lấy tên lớp dự đoán
    return predicted_label, prediction


# Hàm tính toán độ chính xác cho mỗi lớp
def evaluate_model_on_test_data(test_dir, class_labels):
    results = {}

    for class_name in class_labels:
        class_path = os.path.join(test_dir, class_name)
        if not os.path.exists(class_path):
            continue

        correct_predictions = 0
        total_images = 0

        for img_name in os.listdir(class_path):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                img_path = os.path.join(class_path, img_name)

                # Tiền xử lý ảnh
                img_array = preprocess_image(img_path)

                # Dự đoán nhãn
                predicted_label, prediction = predict_image(model, img_array, class_labels)

                # So sánh với nhãn thực tế
                if predicted_label == class_name:
                    correct_predictions += 1

                total_images += 1

        if total_images > 0:
            accuracy = correct_predictions / total_images
            results[class_name] = (correct_predictions, total_images, accuracy)

    return results


# Chạy đánh giá mô hình
evaluation_results = evaluate_model_on_test_data(test_dir, class_labels)

# In kết quả thống kê
for class_name, (correct, total, accuracy) in evaluation_results.items():
    print(f"Class: {class_name}")
    print(f"Correct Predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("-" * 30)


