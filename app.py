import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Botanical Classifier",
    page_icon="🌸",
    layout="centered"
)

# Custom CSS for a "Classic" look (Serif fonts, clean borders)
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    h1 {
        font-family: 'Times New Roman', Times, serif;
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        color: white;
        background-color: #2c3e50;
        border-radius: 5px;
    }
    .uploaded-img {
        border: 5px solid #ddd;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- APP LOGIC ---

@st.cache_resource
def load_model():
    # Load model and class names
    model = tf.keras.models.load_model('flower_model.h5')
    with open("class_names.txt", "r") as f:
        class_names = f.read().splitlines()
    return model, class_names

def predict_flower(image, model, class_names):
    # Preprocess image to match training input (180x180)
    img = image.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = 100 * np.max(predictions[0])
    return predicted_class, confidence

# --- UI LAYOUT ---
st.title("The Botanical Classifier")
st.markdown("---")
st.write("Upload an image of a flower to identify its species. Currently supports: **Daisy, Dandelion, Rose, Sunflower, Tulip**.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("This application uses a Convolutional Neural Network (MobileNetV2) trained on the TensorFlow Flowers dataset.")
    st.write("Created with Python & Streamlit.")

# File Uploader
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Specimen', use_container_width=True)
    
    # Load model (cached)
    with st.spinner('Analyzing specimen details...'):
        model, classes = load_model()
        label, confidence = predict_flower(image, model, classes)

    # Display Result
    st.markdown("---")
    st.markdown(f"<h3 style='text-align: center; color: #1f77b4;'>Identification: {label.upper()}</h3>", unsafe_allow_html=True)
    st.progress(int(confidence))
    st.caption(f"Confidence Level: {confidence:.2f}%")