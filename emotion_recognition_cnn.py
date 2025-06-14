import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Các lớp cảm xúc
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Hàm xây dựng mô hình CNN cho ảnh xám (1 kênh)
def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Huấn luyện mô hình
def train_model():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        train_dir = os.path.join(base_dir, 'train')
        test_dir = os.path.join(base_dir, 'test')

        # Tiền xử lý ảnh xám
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical'
        )

        validation_generator = val_datagen.flow_from_directory(
            test_dir,
            target_size=(48, 48),
            color_mode='grayscale',
            batch_size=32,
            class_mode='categorical'
        )

        model = build_cnn_model()

        history = model.fit(
            train_generator,
            epochs=50,
            validation_data=validation_generator
        )

        model.save(os.path.join(base_dir, 'emotion_model.h5'))
        print("✅ Đã lưu mô hình vào emotion_model.h5")

        # Vẽ biểu đồ Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
        plt.title('Biểu đồ Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện: {str(e)}")

# Chạy
if __name__ == "__main__":
    print("🔍 Bắt đầu huấn luyện mô hình nhận diện cảm xúc (ảnh xám)...")
    train_model()