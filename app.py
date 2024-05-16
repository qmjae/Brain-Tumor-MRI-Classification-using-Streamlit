import streamlit as st
import tensorflow as tf
import PIL
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('fmodel.h5')
    return model

model = load_model()

# Streamlit UI design
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.header('Group 3 - CPE 019-CPE32S6')
st.sidebar.markdown("Ejercito, Marlon Jason")
st.sidebar.markdown("Flores, Mico Joshua")
st.sidebar.markdown("Flores, Marc Oliver")
st.sidebar.markdown("Gabiano, Chris Leonard")
st.sidebar.markdown("Gomez, Joram")

st.sidebar.header('Github Link')
st.sidebar.markdown("[Click Here](https://github.com/qmjae/Brain-Tumor-MRI-Classification-using-Streamlit)")

st.sidebar.header('Google Drive Link')
st.sidebar.markdown("[Click Here](https://drive.google.com/drive/folders/1MExGDFt6MVJunB97RloUM7sNb3rudecz?usp=sharing)")

st.sidebar.header('Google Colaboratory Link')
st.sidebar.markdown("[Click Here](https://colab.research.google.com/drive/1voRF5tQ49C45BU7mJRV8wBKjji_YANz5?usp=sharing)")

st.markdown(
    """
    <style>
    /* Add a border around the title */
    .title-container {
        border: 2px solid #f63366;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title-container'><h1>Brain Tumor MRI Classification</h1></div>", unsafe_allow_html=True)

file = st.file_uploader("Choose a Brain MRI image", type=["jpg", "png", "jpeg"])

def import_and_predict(image_data, model):
    size = (150, 150)  # Match the input size with the Google Colab code
    image = ImageOps.fit(image_data, size, PIL.Image.LANCZOS)  # Use PIL.Image.LANCZOS for resizing
    image = image.convert('RGB')  # Ensure image has 3 color channels (RGB)
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    try:
        image = Image.open(file)
        st.image(image, use_column_width=True, output_format='JPEG')
        
        # Add a border to the image
        st.markdown(
            "<style> img { display: block; margin-left: auto; margin-right: auto; border: 2px solid #ccc; border-radius: 8px; } </style>",
            unsafe_allow_html=True
        )
        
        prediction = import_and_predict(image, model)
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        
        # Display confidence levels
        confidence_levels = {class_name: round(float(pred), 4) for class_name, pred in zip(class_names, prediction[0])}
        st.write("Confidence levels:")
        for class_name, confidence in confidence_levels.items():
            st.write(f"{class_name}: {confidence * 100:.2f}%")
        
        # Display the most likely class
        string = "OUTPUT : " + class_names[np.argmax(prediction)]
        st.success(string)
    except Exception as e:
        st.error("Error: Please upload an image file with one of the following formats: .JPG, .PNG, or .JPEG")
