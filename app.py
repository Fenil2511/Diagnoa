import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the adversarially trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("adversarial_trained_model.h5")
    return model

model = load_model()

# UI
st.title("Pneumonia Detection (Adversarially Trained Model)")
st.write("Upload a Chest X-ray image (150x150)...")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((150, 150))
    
    # Display image
    st.image(image_resized, caption="Uploaded X-ray", use_column_width=True)
    
    # Preprocess
    img_array = np.array(image_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 150, 150, 3)

    # Predict
    preds = model.predict(img_batch)
    class_names = ['Normal', 'Pneumonia']
    predicted_class = class_names[np.argmax(preds)]

    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {np.max(preds)*100:.2f}%")
