# Prediction and siplay the image with the predicted label in streamlit
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Digit Sign Language Recognition')
st.text('Upload a digit sign language image for image classification as 0-9')

model = tf.keras.models.load_model('model/digitSignLanguage.h5')

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
        size = (32, 32)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    string = "This image most likely is: " + class_names[np.argmax(predictions)]
    st.success(string)
    st.text("Probability (0-9):")
    st.write(predictions)

#streamlit run app.py
    
