import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- LOAD MODEL ---
model = load_model('dog_breed_model.h5')

# --- CLASS LABELS (update to match your dataset) ---
class_names = ['german_shepherd', 'golden_retriever', 'poodle', 'french_bulldog', 'yorkshire_terrier']

# --- STREAMLIT APP ---
st.title("🐶 Dog Breed Classifier")
st.write("Upload an image of a dog to predict its breed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}")
