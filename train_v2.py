

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow.keras.mixed_precision import set_global_policy

# Thiết lập mixed precision
set_global_policy('mixed_float16')

# Đường dẫn tới tập dữ liệu
dataset_dir = r'D:\PBL4\Project\pbl4\datasets-BKĐN\Training'

# Thiết lập ImageDataGenerator với các kỹ thuật augment hợp lệ
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],  # Điều chỉnh độ sáng
    channel_shift_range=0.1,  # Dịch kênh màu
    fill_mode='nearest'
)

# Tạo generator cho tập huấn luyện và kiểm tra
train_generator = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(224, 224),  # Kích thước ảnh chuẩn của EfficientNetB0
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Lấy số lượng lớp
num_classes = train_generator.num_classes

# Sử dụng mô hình pre-trained EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Đóng băng các lớp pre-trained (ngoại trừ 20 lớp cuối cùng)
for layer in base_model.layers[:-20]:  # Mở khóa 20 lớp cuối cùng để fine-tune
    layer.trainable = False

# Xây dựng mô hình với Attention Mechanism
inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

# Thêm cơ chế Attention
attention_input = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)  # Mở rộng chiều
attention_output = MultiHeadAttention(num_heads=8, key_dim=512)(attention_input, attention_input)

# Sử dụng Lambda layer để xóa chiều kích thước 1
attention_output_squeezed = layers.Lambda(lambda x: tf.squeeze(x, axis=1))(attention_output)

# Kết hợp đầu ra của Attention
x = layers.Add()([x, attention_output_squeezed])  # Kết hợp đầu ra

# Tiếp tục xây dựng mô hình với các lớp Dense và Dropout
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)  # Giữ nguyên softmax

# Tạo mô hình hoàn chỉnh
model = models.Model(inputs, outputs)

# Sử dụng Adam optimizer với điều chỉnh learning rate
optimizer = Adam(learning_rate=0.001, weight_decay=1e-5)

# Sử dụng Label Smoothing trong hàm loss
loss_fn = CategoricalCrossentropy(label_smoothing=0.1)

# Biên dịch mô hình với optimizer Adam và mixed precision
model.compile(
    loss=loss_fn,
    optimizer=optimizer,
    metrics=['accuracy']
)

# Thiết lập các callbacks
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Lưu mô hình
model.save('final_model_advanced.h5')

# Vẽ biểu đồ accuracy và loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

