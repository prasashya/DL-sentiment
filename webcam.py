import cv2
import numpy as np
from tensorflow.keras.models import load_model
from load import load_dataset

# Load your saved model
model = load_model("emotion_cnn_model.h5")  
_, _, le = load_dataset()  

# Map emotion to sentiment
def emotion_to_sentiment(emotion_label):
    if emotion_label in ['happy','surprise']:
        return "Positive"
    elif emotion_label == 'neutral':
        return "Neutral"
    else:
        return "Negative"

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face.reshape(1,48,48,1)/255.0

        pred = model.predict(face)
        emotion_label = le.inverse_transform([np.argmax(pred)])[0]
        sentiment = emotion_to_sentiment(emotion_label)

        # Draw rectangle and label
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, f"{emotion_label} ({sentiment})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Webcam Sentiment Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
