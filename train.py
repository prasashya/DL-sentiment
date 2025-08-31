from load import load_dataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
images, labels_cat, le = load_dataset()

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    images, labels_cat, test_size=0.1, random_state=42
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Train the model using augmented training data
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(X_val, y_val)
)
model.save("emotion_cnn_model.h5")

# Function to map emotion to sentiment
def emotion_to_sentiment(emotion_label):
    if emotion_label in ['happy','surprise']:
        return "Positive"
    elif emotion_label == 'neutral':
        return "Neutral"
    else:
        return "Negative"

# Test on a single image
pred = model.predict(images[0:1])
emotion_label = le.inverse_transform([np.argmax(pred)])[0]
sentiment = emotion_to_sentiment(emotion_label)
print("Predicted sentiment:", sentiment)
