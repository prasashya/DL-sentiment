import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

def load_dataset(dataset_path=r"C:\Users\prasa\Desktop\DL-sentiment\dataset"):
    images = []
    labels = []

    for split in os.listdir(dataset_path):  # train/test
        split_path = os.path.join(dataset_path, split)
        if not os.path.isdir(split_path):
            continue
        for emotion_folder in os.listdir(split_path):
            folder_path = os.path.join(split_path, emotion_folder)
            if not os.path.isdir(folder_path):
                continue
            img_files = os.listdir(folder_path)
            print(f"{split}/{emotion_folder} folder has {len(img_files)} items")

            for img_name in img_files:
                img_path = os.path.join(folder_path, img_name)
                if not os.path.isfile(img_path) or not img_name.lower().endswith(('.png','.jpg','.jpeg')):
                    continue
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print("Warning: could not read", img_path)
                    continue
                img = cv2.resize(img, (48,48))
                images.append(img)
                labels.append(emotion_folder)

    print(f"Total images loaded: {len(images)}")

    if len(images) == 0:
        raise ValueError("No images loaded! Check dataset path and image extensions.")

    images = np.array(images).reshape(-1,48,48,1)/255.0
    labels = np.array(labels)
    le = LabelEncoder()
    labels_num = le.fit_transform(labels)
    labels_cat = to_categorical(labels_num)

    return images, labels_cat, le
