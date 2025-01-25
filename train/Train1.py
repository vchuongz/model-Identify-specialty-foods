import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import cv2

# Đường dẫn tới dataset
dataset_dir = r'D:\PBL4\Project\pbl4\datasets-BKĐN\Training'


# Hàm áp dụng cả 3 loại blur
def apply_all_blurs(image):
    image = np.array(image)
    # Gaussian Blur
    blurred_gaussian = cv2.GaussianBlur(image, (7, 7), 0)
    # Median Blur
    blurred_median = cv2.medianBlur(image, 5)
    # Bilateral Filter
    blurred_bilateral = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    return [blurred_gaussian, blurred_median, blurred_bilateral]

# Generator tùy chỉnh để nhân dataset với 3 loại blur
def custom_data_generator(generator, apply_blurs_func):
    for batch_images, batch_labels in generator:
        augmented_images = []
        augmented_labels = []

        for image, label in zip(batch_images, batch_labels):
            # Áp dụng cả 3 loại blur cho mỗi ảnh
            blurred_images = apply_blurs_func(image)
            augmented_images.extend(blurred_images)
            augmented_labels.extend([label] * len(blurred_images))

        yield np.array(augmented_images), np.array(augmented_labels)


# ImageDataGenerator cơ bản
#chia kt huan luyen
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

# Tạo generator cơ bản
#truy cap sau 32
#cste mahoa
base_train_generator = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    seed=42
)

base_val_generator = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    seed=42
)

#200 mẫu trong base_val_generator, sau khi áp dụng 3 loại làm mờ, bạn sẽ có tổng cộng 600 mẫu. Với batch_size=32, số bước mỗi epoch sẽ là 600 // 32 = 18 bước.
# Generator áp dụng làm mờ
train_generator = custom_data_generator(base_train_generator, apply_all_blurs)
validation_generator = custom_data_generator(base_val_generator, apply_all_blurs)

# Số bước mỗi epoch
steps_per_epoch = (base_train_generator.samples * 3) // base_train_generator.batch_size
validation_steps = (base_val_generator.samples * 3) // base_val_generator.batch_size


# Xây dựng mô hình CNN
def build_model(dropout_rate=0.5):
    model = models.Sequential([
        layers.Input(shape=(150, 150, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),#cai nay ve 1 chiu
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(base_train_generator.num_classes, activation='softmax')#classnum
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model


# Hàm huấn luyện và tối ưu hóa mô hình
def optimize_and_train(max_epochs=20, target_mse=0.2, target_acc=0.7):
    best_mse = float('inf')
    best_acc = 0.0
    best_model = None
    best_history = None  # Lưu lịch sử huấn luyện tốt nhất

    for dropout_rate in [0.3, 0.5, 0.7]:
        print(f"Training with parameters: {{'dropout_rate': {dropout_rate}}}")
        model = build_model(dropout_rate)
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
            ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
        ]

        # Huấn luyện mô hình
        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            epochs=max_epochs,
            callbacks=callbacks
        )

        # Đánh giá mô hình
        val_loss, val_acc = model.evaluate(validation_generator, steps=validation_steps)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        # Cập nhật mô hình tốt nhất
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
            best_history = history

        # Kiểm tra nếu đạt được mục tiêu
        if val_acc >= target_acc:
            print(f"Target reached: Accuracy={val_acc:.4f}")
            break

    # Lưu mô hình tốt nhất
    if best_model:
        best_model.save('windsosad.h5')
    else:
        print("No model met the target criteria.")

    return best_model, best_acc, best_history


# Huấn luyện mô hình
best_model, best_acc, best_history = optimize_and_train()


# Hàm vẽ biểu đồ lịch sử huấn luyện
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    # Vẽ biểu đồ độ chính xác
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')

    # Vẽ biểu đồ loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


# Vẽ biểu đồ lịch sử huấn luyện nếu mô hình tốt nhất được tìm thấy
if best_history:
    plot_training_history(best_history)
else:
    print("No model met the target criteria.")
