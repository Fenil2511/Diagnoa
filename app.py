import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Model URLs (Google Drive file IDs)
model_urls = {
    "Normal Model": {
        "filename": "normal_trained_model.h5",
        "file_id": "1YeCFfSnUT09w5Ihd2Uq3tka2is2VqJGd"  # Replace with actual file ID
    },
    "Adversarially Trained Model": {
        "filename": "adversarial_trained_model.h5",
        "file_id": "1OUvXDLFgd5Fe9No-ooJXHDbeMQ_KvHL2"
    }
}

# Streamlit UI
st.title("🩻 Pneumonia Detection with Adversarial Defense")
st.write("Upload a Chest X-ray (150x150), and select which model you'd like to use.")

# Model selection
model_choice = st.selectbox("Choose the model for prediction:", list(model_urls.keys()))

# Download model if not exists
model_file = model_urls[model_choice]['filename']
file_id = model_urls[model_choice]['file_id']
if not os.path.exists(model_file):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_file, quiet=False)

# Load selected model
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(model_file)

# Image uploader
uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((150, 150))
    st.image(image_resized, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img_array = np.array(image_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_batch)
    class_names = ['Normal', 'Pneumonia']
    prediction = class_names[np.argmax(preds)]

    # Output
    st.success(f"Model: **{model_choice}**")
    st.success(f"Prediction: **{prediction}**")
    st.write(f"Confidence: {np.max(preds) * 100:.2f}%")
