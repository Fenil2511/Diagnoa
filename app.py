import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Download model from Google Drive if not exists
model_file = "adversarial_trained_model.h5"
if not os.path.exists(model_file):
    file_id = "1H2yebeR9YtudZTmheeblUaDAQ8Ys38GV"  # Replace with your file ID
    gdown.download(f"https://drive.google.com/uc?id=1H2yebeR9YtudZTmheeblUaDAQ8Ys38GV", model_file, quiet=False)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_file)
    return model

model = load_model()

# UI
st.title("Pneumonia Detection (Adversarially Trained)")
st.write("Upload a Chest X-ray image (150x150)...")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((150, 150))
    st.image(image_resized, caption="Uploaded X-ray", use_column_width=True)

    img_array = np.array(image_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_batch)
    class_names = ['Normal', 'Pneumonia']
    prediction = class_names[np.argmax(preds)]

    st.success(f"Prediction: **{prediction}**")
    st.write(f"Confidence: {np.max(preds) * 100:.2f}%")
