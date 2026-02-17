import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl","rb"))
encoders = pickle.load(open("encoders.pkl","rb"))

st.title("Customer Churn Prediction")

gender = st.selectbox("Gender", ["Male","Female"])
age = st.number_input("Age",18,100)
married = st.selectbox("Married", ["Yes","No"])
dependents = st.number_input("Number of Dependents",0,10)
tenure = st.number_input("Tenure in Months",0,100)
internet = st.selectbox("Internet Service", ["Yes","No"])
contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
payment = st.selectbox("Payment Method",
                       ["Electronic check","Mailed check",
                        "Bank transfer","Credit card"])
monthly = st.number_input("Monthly Charge")
total = st.number_input("Total Charges")

# Encode
gender = encoders['Gender'].transform([gender])[0]
married = encoders['Married'].transform([married])[0]
internet = encoders['Internet Service'].transform([internet])[0]
contract = encoders['Contract'].transform([contract])[0]
payment = encoders['Payment Method'].transform([payment])[0]

input_data = np.array([[gender, age, married,
                        dependents, tenure,
                        internet, contract,
                        payment, monthly, total]])

if st.button("Predict"):
    pred = model.predict(input_data)

    if pred[0]==0:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")
