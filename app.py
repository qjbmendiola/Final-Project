import streamlit as st
import numpy as np
from PIL import Image

# Removed all custom CSS and the st.markdown style block

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

# --- Breed List ---
class_names = ['French Bulldog', 'German Shepherd', 'Golden Retriever', 'Poodle', 'Yorkshire Terrier']
# ------------------

# --- Page Header and Introduction ---
st.title('🐶 Dog Breed Classifier')
st.write('Upload an image of a dog to predict its breed.')

# --- Explanation of Breeds ---
st.subheader('Supported Breeds')
st.info(
    f"""
    This model is trained to classify images into **5 specific dog breeds**:
    - **{class_names[0]}**
    - **{class_names[1]}**
    - **{class_names[2]}**
    - **{class_names[3]}**
    - **{class_names[4]}**
    
    For the best results, please upload an image of one of these breeds!
    """
)
# ----------------------------------

if model is None or not MODEL_IMPORTS_SUCCESS:
    st.info("App functionality is disabled due to critical dependency failure.")
else:
    # Single column layout for maximum simplicity
    
    st.subheader('1. Upload an Image')
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        st.subheader('2. Image & Prediction')
        st.image(img, caption='Uploaded Photo', use_container_width=True) 

        with st.spinner('Analyzing image and predicting breed...'):
            
            # --- Preprocessing identical to original code ---
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            img_array = img_array.astype(np.float32)
            
            try:
                predictions = model.predict(img_array)
            except Exception as e:
                st.error(f"Error during prediction: {e}. Please check model compatibility, shape, and normalization.")
                st.stop()
                
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            
        # --- Simple Streamlit Results Display ---
        st.success(f'Prediction Confirmed: {predicted_class}')
        st.header('Prediction Results')

        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        
        cols_results = st.columns(3)
        
        for i, idx in enumerate(top_3_indices):
            breed = class_names[idx]
            score = predictions[0][idx]
            
            with cols_results[i]:
                # Use st.metric for a clean display of results
                if i == 0:
                    st.metric(label=f"🥇 Predicted Breed", value=breed, delta=f"{score*100:.2f}% Confidence")
                else:
                    st.metric(label=f"{i+1}. Alternative", value=breed, delta=f"{score*100:.2f}% Confidence")
        # ------------------------------------------------

    else:
        st.info("Waiting for an image to be uploaded...")
        st.image(
            "https://placehold.co/400x300/e0e0e0/333333?text=Image+Preview",
            caption="Preview Area",
            use_container_width=True
        )
