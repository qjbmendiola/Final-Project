import streamlit as st
import numpy as np
from PIL import Image

try:
    # Try importing from the older Keras path
    from keras.models import load_model
    from keras.preprocessing import image
    MODEL_IMPORTS_SUCCESS = True
except ImportError:
    try:
        # Try importing from the TensorFlow Keras path (recommended modern path)
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing import image
        MODEL_IMPORTS_SUCCESS = True
    except ImportError:
        # Fallback error state if both fail
        st.error("🚨 Module Error: The 'keras' or 'tensorflow' dependency failed to install/load.")
        st.warning("Please verify your Requirements.txt and ensure a clean Streamlit redeploy.")
        MODEL_IMPORTS_SUCCESS = False

@st.cache_resource
def load_classification_model():
    try:
        # Ensure the model file is present in the repository root
        return load_model('dog_breed_model.h5')
    except Exception as e:
        st.error(f"Failed to load model file 'dog_breed_model.h5'. Error: {e}")
        return None

model = None
if MODEL_IMPORTS_SUCCESS:
    model = load_classification_model()

# --- CRITICAL FIX: The order now matches the Colab output (Index 0: French Bulldog, Index 1: German Shepherd, etc.) ---
class_names = ['French Bulldog', 'German Shepherd', 'Golden Retriever', 'Poodle', 'Yorkshire Terrier']
# -----------------------------------------------------------------------------------------------------------------------

st.title('🐶 Dog Breed Classifier')
st.write('Upload an image of a dog to predict its breed.')

if model is None or not MODEL_IMPORTS_SUCCESS:
    st.info("App functionality is disabled due to critical dependency failure.")
else:
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        # NOTE: Using 'use_container_width' instead of deprecated 'use_column_width'
        st.image(img, caption='Uploaded Image', use_container_width=True) 

        with st.spinner('Analyzing image and predicting breed...'):
            
            # --- CRITICAL FIX: Unconditionally convert to RGB (3 channels) first to prevent 1-channel or 4-channel errors ---
            img = img.convert('RGB')
            # -----------------------------------------------------------------------------------------------------------------

            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            # --- Previous FIX: Explicitly cast to float32 to match model's expected dtype ---
            img_array = img_array.astype(np.float32)
            # ------------------------------------------------------------------------

            try:
                predictions = model.predict(img_array)
            except Exception as e:
                st.error(f"Error during prediction: {e}. Please check model compatibility, shape, and normalization.")
                # Exit the block if prediction failed
                st.stop()
                
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
