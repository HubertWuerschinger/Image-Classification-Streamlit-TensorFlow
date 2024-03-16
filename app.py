import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Es ist nicht mehr notwendig, die folgende Option zu setzen, da sie veraltet ist.
# st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
    # Laden Sie das Modell ohne die Verlustfunktion und kompilieren Sie es erneut.
    model = tf.keras.models.load_model('./flower_model_trained.hdf5', compile=False)
    # Definieren Sie hier Ihre Verlustfunktion, Optimizer und Metriken
    # Beispiel: 
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def predict_class(image, model):
    image = Image.open(image).convert('RGB')  # Konvertieren in RGB
    image = image.resize((180, 180))          # Bildgröße anpassen
    image_array = np.array(image)             # In ein Numpy-Array konvertieren
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array) # Bild vorverarbeiten
    image_array = np.expand_dims(image_array, axis=0)  # Dimension erweitern

    prediction = model.predict(image_array)  # Vorhersage durchführen
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
