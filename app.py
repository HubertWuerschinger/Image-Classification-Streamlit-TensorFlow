import streamlit as st
import tensorflow as tf
import h5py  # Für HDF5-Version
from PIL import Image
import numpy as np

# Zeige TensorFlow und HDF5-Versionen an
st.text(f"TensorFlow Version: {tf.__version__}")
st.text(f"HDF5 Version: {h5py.__version__}")

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('./flower_model_trained.hdf5', compile=False)
    return model

def predict_class(image, model):
    image = Image.open(image).convert('RGB')
    image = image.resize((180, 180))
    image_array = np.array(image)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    return prediction

model = load_model()

st.title('Flower Classifier')

file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])

if file is None:
    st.text('Waiting for upload....')
else:
    slot = st.empty()
    slot.text('Running inference....')

    test_image = Image.open(file)
    st.image(test_image, caption="Input Image", width=400)

    pred = predict_class(file, model)

    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    result = class_names[np.argmax(pred)]

    output = 'The image is a ' + result
    slot.text('Done')
    st.success(output)
