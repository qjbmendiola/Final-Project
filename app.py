import streamlit as st
import numpy as np
from PIL import Image

# --- Custom Styling (Creative GUI) ---
st.markdown("""
    <style>
    /* Use a playful, modern font if available in the environment */
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Fredoka', sans-serif;
    }

    /* Main Title Styling */
    .dog-title {
        font-size: 48px !important;
        font-weight: 600;
        color: #ff4500; /* Warm Orange/Red */
        text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Styled Result Card */
    .result-card {
        border-radius: 20px;
        padding: 30px;
        margin-top: 30px;
        background: linear-gradient(135deg, #fffbe0 0%, #f7e8c3 100%); /* Soft gradient background */
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
        border: 3px solid #ff4500;
        text-align: center;
    }
    .prediction-text {
        font-size: 32px;
        font-weight: bold;
        color: #36454F; /* Charcoal */
        margin-top: 10px;
    }

    /* Confidence Bar Styling */
    .confidence-bar-container {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 25px;
        overflow: hidden;
        margin-top: 15px;
    }
    .confidence-fill {
        background-color: #4CAF50; /* Success Green */
        border-radius: 10px;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        color: white;
        font-weight: 600;
        padding-right: 10px;
        font-size: 14px;
        transition: width 0.5s;
    }
    .stSpinner > div {
        color: #ff4500;
    }
    </style>
    """, unsafe_allow_html=True)

# Try to enforce a wide layout if possible (might require config change)
# st.set_page_config(layout="wide") 

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
st.markdown('<p class="dog-title">🐶 Dog Breed Detective 🕵️</p>', unsafe_allow_html=True)

# Create a two-column layout: Column 1 for Input, Column 2 for Output
col_input, col_output = st.columns([1, 2]) # 1 part for input, 2 parts for display/results

with col_input:
    st.subheader('1. Upload an Image')
    st.write('Drop a picture of your canine friend below:')
    
    # File uploader goes into the first column
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    # --- NEW EXPLANATION ADDED HERE ---
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
    with col_output:
        st.info("App functionality is disabled due to critical dependency failure.")
else:
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        with col_output:
            st.subheader('2. Image & Results')
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
                confidence_percent = confidence * 100
                
            # --- Creative Result Card Display (Replaces st.success/st.info) ---
            st.markdown(f"""
                <div class="result-card">
                    <h2 style="color:#ff4500; margin-top:0;">Prediction Confirmed!</h2>
                    <p class="prediction-text">
                        The model believes this is a **{predicted_class}**!
                    </p>
                    
                    <p style="margin-top: 20px;">Model Confidence:</p>
                    <div class="confidence-bar-container">
                        <div class="confidence-fill" style="width: {confidence_percent:.0f}%">
                            {confidence_percent:.2f}%
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            # -----------------------------------------------------------------

            # --- Top 3 Display ---
            st.markdown("---")
            st.subheader('Top Alternatives')
            
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            
            # Use columns for a more organized display of the top 3
            cols_results = st.columns(3)
            
            for i, idx in enumerate(top_3_indices):
                breed = class_names[idx]
                score = predictions[0][idx]
                
                with cols_results[i]:
                    # Highlight the main prediction (it will be the first one)
                    if i == 0:
                        st.markdown(f"**🥇 {breed}**")
                        st.metric(label="Confidence", value=f"{score:.2f}")
                    else:
                        st.markdown(f"**{i+1}. {breed}**")
                        st.metric(label="Confidence", value=f"{score:.2f}")

    else:
        with col_output:
            st.info("Waiting for an image to be uploaded...")
            st.image(
                "https://placehold.co/400x300/e0e0e0/333333?text=Image+Preview",
                caption="Preview Area",
                use_container_width=True
            )
