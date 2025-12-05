import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Page configuration
st.set_page_config(
    page_title="Autism Detection Dashboard",
    page_icon=":brain:",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    import warnings
    warnings.filterwarnings('ignore')
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Load your best working model
    model = tf.keras.models.load_model(
        'BEST_MODEL_acc_0.9518_round_33.h5',
        compile=False
    )
    return model

# Preprocess image - for EfficientNet model
def preprocess_image(image, target_size=(224, 224)):
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Resize image to 224x224
    img_resized = cv2.resize(img_array, target_size)
    
    # Add batch dimension
    img_batch = np.expand_dims(img_resized, axis=0)
    
    # Apply EfficientNet-specific preprocessing
    img_preprocessed = preprocess_input(img_batch)
    
    return img_preprocessed

# Main app
def main():
    st.title("Autism Detection Dashboard")
    st.markdown("---")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please check if the model file exists.")
        return
    
    st.success(f"Model loaded successfully! (Accuracy: 95.18%)")
    
    # File uploader
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image for autism detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Prediction Result")
            
            # Add predict button
            if st.button("Analyze Image", type="primary"):
                with st.spinner("Analyzing..."):
                    # Preprocess and predict
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img, verbose=0)
                    
                    # Get prediction probability (single output model)
                    prob = prediction[0][0]
                    
                    # MODEL OUTPUT IS INVERTED:
                    # Low values (< 0.5) = Autism
                    # High values (> 0.5) = Non-Autism
                    if prob < 0.5:
                        result = "Autism"
                        confidence = min((1 - prob) * 100, 99)  # Real confidence
                        color = "#ff6b6b"
                    else:
                        result = "Non-Autism"
                        confidence = min(prob * 100, 99)  # Real confidence
                        color = "#51cf66"
                    
                    # Display result
                    st.markdown(f"""
                    <div style='padding: 20px; border-radius: 10px; background-color: {color}20; border: 2px solid {color};'>
                        <h2 style='color: {color}; text-align: center; margin: 0;'>{result}</h2>
                        <p style='text-align: center; font-size: 18px; margin-top: 10px;'>
                            Confidence: {confidence:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show probability bar
                    st.markdown("### Probability Distribution")
                    # Invert for display (low prob = autism)
                    display_prob = 1 - prob if prob < 0.5 else prob
                    st.progress(float(display_prob))
                    st.caption(f"Confidence: {confidence:.1f}%")
    
    # Information section
    st.markdown("---")
    st.markdown("""
    ### About
    This dashboard uses a deep learning model to detect autism from images.
    - **Model Accuracy**: 95.18%
    - **Upload Format**: JPG, JPEG, PNG, BMP
    
    **Disclaimer**: This tool is for research purposes only and should not be used as a substitute for professional medical diagnosis.
    """)

if __name__ == "__main__":
    main()
