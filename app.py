import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl","rb"))

st.title("Customer Churn Prediction")

# Example inputs (edit based on your features)
age = st.number_input("Age")
balance = st.number_input("Balance")

if st.button("Predict"):
    input_data = np.array([[age, balance]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer will churn")
    else:
        st.success("Customer will stay")
