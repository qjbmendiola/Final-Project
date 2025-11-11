import streamlit as st
import numpy as np
from PIL import Image

try:
    from keras.models import load_model
    from keras.preprocessing import image
    MODEL_IMPORTS_SUCCESS = True
except ImportError:
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        MODEL_IMPORTS_SUCCESS = True
    except ImportError:
        st.error("🚨 Module Error: The 'keras' or 'tensorflow' dependency failed to install/load.")
        st.warning("Please verify your Requirements.txt and ensure a clean Streamlit redeploy.")
        MODEL_IMPORTS_SUCCESS = False

@st.cache_resource
def load_classification_model():
    try:
        return load_model('dog_breed_model.h5')
    except Exception as e:
        st.error(f"Failed to load model file 'dog_breed_model.h5'. Error: {e}")
        return None

model = None
if MODEL_IMPORTS_SUCCESS:
    model = load_classification_model()

class_names = ['german_shepherd', 'golden_retriever', 'poodle', 'french_bulldog', 'yorkshire_terrier']

st.title('🐶 Dog Breed Classifier')
st.write('Upload an image of a dog to predict its breed.')

if model is None or not MODEL_IMPORTS_SUCCESS:
    st.info("App functionality is disabled due to critical dependency failure.")
else:
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
