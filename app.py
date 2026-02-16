import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl","rb"))

st.title("Customer Churn Prediction")

# Numeric inputs
age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure in Months", 0, 100)
monthly = st.number_input("Monthly Charge")
total = st.number_input("Total Charges")

# Categorical inputs
gender = st.selectbox("Gender", ["Male","Female"])
married = st.selectbox("Married", ["Yes","No"])
internet = st.selectbox("Internet Service", ["Yes","No"])
contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
payment = st.selectbox("Payment Method", ["Credit card","Bank transfer","Electronic check"])

# Encoding
gender = 1 if gender=="Male" else 0
married = 1 if married=="Yes" else 0
internet = 1 if internet=="Yes" else 0

contract = {"Month-to-month":0,"One year":1,"Two year":2}[contract]
payment = {"Credit card":0,"Bank transfer":1,"Electronic check":2}[payment]

if st.button("Predict"):
    input_data = np.array([[age, tenure, monthly, total,
                            gender, married, internet,
                            contract, payment]])

    pred = model.predict(input_data)

    if pred[0]==0:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")
