import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Potato Disease Classifier",
    page_icon="🥔",
    layout="centered"
)

# Custom CSS for an "attractive" green theme
st.markdown("""
<style>
    .main {
        background-color: #f0f8f0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-10trblm {
        color: #2e7d32;
    }
    h1 {
        color: #1b5e20;
        text-align: center;
    }
    .reportview-container {
        background: #f0fdf4;
    }
</style>
""", unsafe_allow_html=True)

# Define constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
MODEL_PATH = r"c:\Users\raiya\OneDrive\Desktop\codes\ml_notebook\tomato disease\New Plant Diseases Dataset(Augmented)\models.keras"
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Load the model
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_trained_model()

def preprocess_image(image: Image.Image):
    """Resizes and converts image to numpy array."""
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create batch axis
    return img_array

def predict(model, img_array):
    """Predicts class and confidence."""
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# App UI
st.title("🥔 Potato Disease Classifier")
st.write("Upload a leaf image or take a photo to detect diseases.")

# Toggle between upload and camera
option = st.radio("Choose Input Method", ("Upload Photo", "Use Camera"))

image = None

if option == "Upload Photo":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file)

if image is not None:
    st.image(image, caption="Current Image", use_container_width=True)
    
    if st.button("Predict"):
        if model is not None:
            with st.spinner("Analyzing..."):
                processed_img = preprocess_image(image)
                label, score = predict(model, processed_img)
                
                # Display results
                if "healthy" in label.lower():
                    st.success(f"Result: {label} (Confidence: {score}%)")
                else:
                    st.warning(f"Result: {label} (Confidence: {score}%)")
                
                # Add some info about the disease
                if "Early_blight" in label:
                    st.info("**Early Blight:** Caused by the fungus *Alternaria solani*. It typically affects older leaves first.")
                elif "Late_blight" in label:
                    st.error("**Late Blight:** Caused by *Phytophthora infestans*. It is a serious disease that can kill plants quickly.")
        else:
            st.error("Model is not loaded. Please check the paths.")

st.divider()
st.caption("Powered by TensorFlow and Streamlit")
