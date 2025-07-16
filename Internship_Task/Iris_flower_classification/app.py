# app.py

import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open('outputs/svm_model.pkl', 'rb'))

# Title
st.title("🌸 Iris Flower Species Classifier")
st.write("Enter the sepal and petal measurements to classify the flower.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    X_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                           columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
    pred = model.predict(X_input)[0]
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    st.success(f"Predicted Species: 🌼 {species_map[pred]}")
