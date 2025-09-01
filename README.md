# Facial Expression Sentiment Classification

This project implements a **Convolutional Neural Network (CNN)** model to classify human facial expressions into sentiment categories (Positive / Negative / Neutral).  
It uses image datasets of faces, preprocesses them with **OpenCV**, and trains a deep learning model to predict the sentiment from input frames, posters, or webcam feed.

---

## 📌 Features
- Face detection and preprocessing with **OpenCV**.  
- CNN-based model for sentiment classification.  
- Training pipeline with validation support.  
- Real-time webcam testing for live sentiment prediction.  
- Modular codebase:
  - `load.py` → dataset loading & preprocessing  
  - `train.py` → training loop for CNN model  
  - `image.py` → run predictions on static images/posters  
  - `webcam.py` → live testing using webcam  

---
# Main libraries used:

- Python 3.x

- TensorFlow / Keras

- OpenCV

- NumPy
  
## ⚙️ Requirements
Install the dependencies before running:

```bash
pip install tensorflow keras scikit-learn opencv-python numpy

```

# Datasets
This model can be trained on:

- FER-2013

- Custom face expression datasets

Ensure images are preprocessed into grayscale and resized (e.g., 48×48).

# Model Architecture

- Conv2D layers with 3×3 filters (feature extraction).

- ReLU activation (non-linearity).

- MaxPooling layers (dimensionality reduction).

- Fully connected layers for final classification.

- Softmax output for sentiment categories.

# Results

- Training accuracy: ~57%

- Validation accuracy: ~60% (baseline model).

- Real-time prediction works for simple test cases

# Future Work

- Improve accuracy with deeper CNN / ResNet / Transformer models.

- Train on larger datasets like AffectNet.

- Explore valence-arousal regression instead of categorical classification.

# Acknowledgments

- Keras Documentation

- OpenCV

- Public facial expression datasets (FER-2013, AffectNet)


