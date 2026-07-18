import streamlit as st
import joblib
import numpy as np
import os

model_path = os.path.join(os.path.dirname(__file__), "breast_cancer_model.pkl")
model = joblib.load(model_path)

st.title("Breast Cancer Prediction")

feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",

    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",

    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

inputs = []
for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)

if st.button("Predict"):

    input_data = np.array(inputs).reshape(1, 30)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Result: Benign (Non-Cancerous Tumor)")
    else:
        st.error("Result: Malignant (Cancerous Tumor)")
