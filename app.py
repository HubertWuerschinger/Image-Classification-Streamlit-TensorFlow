import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set Streamlit configuration
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    # Load the Keras model
    model = tf.keras.models.load_model('./flower_model_trained.hdf5')
    return model

def predict_class(image, model):
    # Preprocess the image
    image = Image.open(image).convert('RGB') # Convert to RGB in case it's a different mode
    image = image.resize((180, 180))         # Resize the image
    image_array = np.array(image)            # Convert image to numpy array
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array) # Preprocess input
    image_array = np.expand_dims(image_array, axis=0)

    # Make the prediction
    prediction = model.predict(image_array)
    return prediction

# Load the model
model = load_model()

# Streamlit UI code
st.title('Flower Classifier')

file = st.file_uploader("Upload an image of a flower", type=["jpg", "png"])

if file is None:
    st.text('Waiting for upload....')
else:
    slot = st.empty()
    slot.text('Running inference....')

    # Display the uploaded image
    test_image = Image.open(file)
    st.image(test_image, caption="Input Image", width=400)

    # Make a prediction
    pred = predict_class(file, model)

    # Define class names
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    # Get the predicted class
    result = class_names[np.argmax(pred)]

    # Display the result
    output = 'The image is a ' + result
    slot.text('Done')
    st.success(output)
