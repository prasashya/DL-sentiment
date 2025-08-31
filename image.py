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

# Load image
image_path = r"C:\Users\prasa\Desktop\DL-sentiment\faces\surprise.jpg"  
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (48,48))
img = img.reshape(1,48,48,1)/255.0

# Predict emotion
pred = model.predict(img)
emotion_label = le.inverse_transform([np.argmax(pred)])[0]
sentiment = emotion_to_sentiment(emotion_label)

print("Predicted emotion:", emotion_label)
print("Predicted sentiment:", sentiment)
