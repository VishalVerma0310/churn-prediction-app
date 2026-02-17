import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl","rb"))

st.title("Customer Churn Prediction")

# ===== INPUTS =====

gender = st.selectbox("Gender", ["Male","Female"])
age = st.number_input("Age",18,100)

married = st.selectbox("Married", ["Yes","No"])
dependents = st.number_input("Number of Dependents",0,10)

tenure = st.number_input("Tenure in Months",0,120)

internet = st.selectbox("Internet Service", ["Yes","No"])

contract = st.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"]
)

payment = st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer","Credit card"]
)

monthly = st.number_input("Monthly Charge",0.0,500.0)
total = st.number_input("Total Charges",0.0,10000.0)

# ===== MANUAL ENCODING =====

gender = 1 if gender=="Male" else 0
married = 1 if married=="Yes" else 0
internet = 1 if internet=="Yes" else 0

contract_map = {
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}
contract = contract_map[contract]

payment_map = {
    "Electronic check":0,
    "Mailed check":1,
    "Bank transfer":2,
    "Credit card":3
}
payment = payment_map[payment]

# ===== PREDICTION =====

input_data = np.array([[gender, age, married,
                        dependents, tenure,
                        internet, contract,
                        payment, monthly, total]])

if st.button("Predict"):

    prediction = model.predict(input_data)

    if prediction[0]==0:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")
