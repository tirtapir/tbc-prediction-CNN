import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io

# Load the model
loaded_model = load_model('tb_cnn_model.h5')

# Streamlit web app
st.title("Tuberculosis Detection using CNN")
st.write("Upload a chest X-ray image to predict Tuberculosis")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img = image.resize((150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = loaded_model.predict(img_array)
    if prediction[0][0] > 0.5:
        st.write("The model predicts this image as: **Positive** for Tuberculosis")
    else:
        st.write("The model predicts this image as: **Normal**")
