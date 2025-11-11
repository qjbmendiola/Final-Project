import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- LOAD MODEL ---
@st.cache_resource
def load_classification_model():
    return load_model('dog_breed_model.h5')

model = load_classification_model()

# --- CLASS LABELS ---
class_names = ['german_shepherd', 'golden_retriever', 'poodle', 'french_bulldog', 'yorkshire_terrier']

# --- STREAMLIT APP ---
st.title('🐶 Dog Breed Classifier')
st.write('Upload an image of a dog to predict its breed.')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    with st.spinner('Analyzing image and predicting breed...'):
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

    st.success(f'**Prediction:** {predicted_class}')
    st.info(f'**Confidence:** {confidence:.2f}')

    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    st.subheader('Top 3 Predictions')
    
    results = []
    for i in top_3_indices:
        results.append(f"- **{class_names[i]}** ({predictions[0][i]:.2f} confidence)")
        
    st.markdown('\n'.join(results))
