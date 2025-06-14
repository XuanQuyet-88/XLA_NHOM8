import cv2
import numpy as np
import tensorflow as tf
import os

# Danh sách các lớp cảm xúc
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def detect_emotion_realtime(model_path='emotion_model.h5'):
    try:
        # Kiểm tra model tồn tại
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")

        # Load mô hình đã huấn luyện
        model = tf.keras.models.load_model(model_path)

        # Tải bộ cascade nhận diện khuôn mặt
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Mở webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("❌ Không thể mở webcam!")

        print("🎥 Đang khởi động webcam. Nhấn 'q' để thoát.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Chuyển sang ảnh xám
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Nhận diện khuôn mặt
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)       # (1, 48, 48)
                face = np.expand_dims(face, axis=-1)      # (1, 48, 48, 1)

                # Dự đoán cảm xúc
                prediction = model.predict(face, verbose=0)
                emotion_label = EMOTIONS[np.argmax(prediction)]
                confidence = np.max(prediction)

                # Vẽ khung và nhãn
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{emotion_label} ({confidence*100:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # Hiển thị khung hình
            cv2.imshow("Real-Time Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    print("🤖 Đang chạy nhận diện cảm xúc real-time...")
    detect_emotion_realtime()
