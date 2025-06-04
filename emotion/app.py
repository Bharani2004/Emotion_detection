from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pygame
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Initialize pygame mixer
pygame.mixer.init()

# Load model and face cascade
try:
    model = load_model("../model/emotion_model.h5")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_songs = {
    "Angry": "songs/angry.mp3",
    "Disgust": "songs/disgust.mp3",
    "Fear": "songs/fear.mp3",
    "Happy": "songs/happy.mp3",
    "Sad": "songs/sad.mp3",
    "Surprise": "songs/surprise.mp3"
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

last_emotion = None
last_detection_time = time.time()
detection_delay = 3

def generate_frames():
    global last_emotion, last_detection_time
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        current_time = time.time()
        
        if current_time - last_detection_time >= detection_delay:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi_gray = roi_gray.astype("float") / 255.0
                roi_gray = img_to_array(roi_gray)
                roi_gray = np.expand_dims(roi_gray, axis=0)

                predictions = model.predict(roi_gray)
                emotion_label = emotion_labels[np.argmax(predictions)]
                confidence = np.max(predictions)

                # Emit emotion data to frontend via WebSocket
                socketio.emit('emotion_update', {
                    'emotion': emotion_label,
                    'confidence': float(confidence)
                })

                # Play song if emotion changes
                if emotion_label != last_emotion:
                    song_path = emotion_songs.get(emotion_label)
                    if song_path:
                        pygame.mixer.music.load(song_path)
                        pygame.mixer.music.play()
                    last_emotion = emotion_label

                # Draw rectangle and text on video feed
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion_label} ({confidence:.2f})", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                last_detection_time = current_time

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield frame in byte format for video streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)