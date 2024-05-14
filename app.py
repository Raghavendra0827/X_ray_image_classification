import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("Xray_model.h5")

# Class labels
class_labels = ['NORMAL', 'PNEUMONIA']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(225, 225))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def main():
    # Add background image using CSS
    st.markdown(
        """
        <style>
        .container {
            background-image: url("https://builtin.com/sites/www.builtin.com/files/styles/og/public/2022-06/ai-healthcare-examples.png");
            background-size: cover;
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='color:Turquoise'>X-Ray Image Classifier</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an X-ray image ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)
        if st.button("Classify"):
            with st.spinner('Classifying...'):
                image = preprocess_image(uploaded_file)
                prediction = model.predict(image)
                predicted_class_index = np.argmax(prediction)
                predicted_class_label = class_labels[predicted_class_index]
            st.success(f"Prediction: **{predicted_class_label}**")

if __name__ == "__main__":
    main()
