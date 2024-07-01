import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import io

# # Define directories
# base_dir = '/Users/tirtarumy/Documents/Data scientist porto/Tuberculosis Classification using MLP CNN/Data/TB'
# normal = os.path.join(base_dir, 'Normal')
# positif = os.path.join(base_dir, 'Positif')
# train_normal = os.listdir(normal)
# train_positif = os.listdir(positif)

# # Preprocessing
# train_dataGen = ImageDataGenerator(
#     validation_split=0.2,
#     rescale=1./255,
#     rotation_range=5,
#     horizontal_flip=True,
#     vertical_flip=True,
#     fill_mode='nearest'
# )

# train_generator = train_dataGen.flow_from_directory(
#     base_dir,
#     target_size=(150, 150),
#     shuffle=True,
#     batch_size=8,
#     subset='training',
#     class_mode='binary'
# )

# validation_generator = train_dataGen.flow_from_directory(
#     base_dir,
#     target_size=(150, 150),
#     shuffle=True,
#     batch_size=8,
#     subset='validation',
#     class_mode='binary'
# )

# # Building the CNN model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# # Early stopping callback
# class my_callback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         if logs.get('accuracy') > 0.9:
#             print('\nFor epoch', epoch,
#                   '\nAccuracy has reached %2.2f%%' % (logs['accuracy']*100),
#                   'training has been stopped')
#             self.model.stop_training = True

# # Training the model
# callbacks = my_callback()
# model.fit(train_generator,
#           epochs=20,
#           validation_data=validation_generator,
#           callbacks=[callbacks])

# # Save the model
# model.save('tb_cnn_model.h5')

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
