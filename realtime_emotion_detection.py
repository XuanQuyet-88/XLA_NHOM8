import os
import cv2
import numpy as np
import tensorflow as tf

# Danh sách các lớp cảm xúc
EMOTIONS = ['tuc gian', 'kho chiu', 'so hai', 'vui ve', 'binh thuong', 'buon', 'ngac nhien']

# Nhận diện cảm xúc real-time từ webcam
def detect_emotion_realtime(model_path='emotion_model.h5'):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")

        model = tf.keras.models.load_model(model_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("❌ Không thể mở webcam!")

        print("🎥 Đang nhận diện cảm xúc real-time. Nhấn 'q' để thoát.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = face.astype('float32') / 255.0
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1)

                prediction = model.predict(face, verbose=0)

                percentages = (prediction[0] * 100).tolist()
                emotion_label = EMOTIONS[np.argmax(prediction)]

                # Hiển thị tất cả cảm xúc bên trái màn hình
                start_y = 20
                for i, (emotion, pct) in enumerate(zip(EMOTIONS, percentages)):
                    color = (0, 0, 255) if emotion == emotion_label else (255, 255, 255)
                    text = f"{emotion}: {pct:.2f}%"
                    cv2.putText(frame, text, (10, start_y + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Vẽ khung và nhãn chính trên khuôn mặt
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"{emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Real-Time Emotion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Lỗi: {str(e)}")

# Nhận diện cảm xúc từ ảnh tĩnh và hiển thị % tất cả cảm xúc
def detect_emotion_from_image(image_path, model_path='emotion_model.h5'):
    import os
    import cv2
    import numpy as np
    import tensorflow as tf

    # Danh sách các lớp cảm xúc
    EMOTIONS = ['tuc gian', 'kho chiu', 'so hai', 'vui ve', 'binh thuong', 'buon', 'ngac nhien']

    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Không tìm thấy mô hình tại: {model_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh tại: {image_path}")

        model = tf.keras.models.load_model(model_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        prediction_done = False
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            prediction = model.predict(face, verbose=0)
            emotion_label = EMOTIONS[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Vẽ khung và nhãn chính
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, f"{emotion_label}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            prediction_done = True
            break  # Chỉ xử lý khuôn mặt đầu tiên

        # Nếu có dự đoán thì vẽ % cảm xúc lên góc trái
        if prediction_done:
            percentages = (prediction[0] * 100).tolist()
            start_y = 30
            for i, (emotion, pct) in enumerate(zip(EMOTIONS, percentages)):
                color = (0, 0, 255) if emotion == emotion_label else (255, 255, 255)
                text = f"{emotion}: {pct:.2f}%"
                cv2.putText(image, text, (10, start_y + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Resize ảnh nếu quá lớn
        max_width = 1000
        h, w = image.shape[:2]
        if w > max_width:
            scale = max_width / w
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # Hiển thị ảnh kết quả
        cv2.imshow("Emotion Detection from Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Lỗi: {str(e)}")


# Chọn chế độ chạy
if __name__ == "__main__":
    print("📌 Chọn chế độ:")
    print("1. Nhận diện cảm xúc real-time (webcam)")
    print("2. Nhận diện cảm xúc từ ảnh")
    choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()

    if choice == "1":
        detect_emotion_realtime()
    elif choice == "2":
        image_path = input("Nhập đường dẫn đến ảnh: ").strip()
        detect_emotion_from_image(image_path)
    else:
        print("❌ Lựa chọn không hợp lệ!")
