import streamlit as st
import joblib
import numpy as np
model=joblib.load("breast_cancer_model.pkl")
st.title("enter the following values:")
mean_radius=st.number_input("Mean Radius",value=14.0)
mean_texture=st.number_input("Mean Texture",value=20.0)
mean_perimeter=st.number_input("Mean Perimeter",value=90.0)
mean_area = st.number_input("Mean Area", value=600.0)
mean_smoothness = st.number_input("Mean Smoothness", value=0.1)

if st.button("Predict"):
    input_data = np.zeros((1, 30))
    input_data[0][0] = mean_radius
    input_data[0][1] = mean_texture
    input_data[0][2] = mean_perimeter
    input_data[0][3] = mean_area
    input_data[0][4] = mean_smoothness

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Result: Benign (Non-Cancerous Tumor)")
    else:
        st.error("Result: Malignant (Cancerous Tumor)")
