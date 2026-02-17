import streamlit as st
import numpy as np
import pickle

# ===== PAGE SETTINGS =====
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# ===== LOAD MODEL =====
model = pickle.load(open("model.pkl","rb"))

# ===== HEADER =====
st.markdown("## 📊 Customer Churn Prediction App")
st.markdown("Predict whether a telecom customer will churn.")

st.markdown("---")

# ===== INPUT SECTION =====

gender = st.selectbox("Gender", ["Male","Female"])
married = st.selectbox("Married", ["Yes","No"])
internet = st.selectbox("Internet Service", ["Yes","No"])

contract = st.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"]
)

payment = st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer","Credit card"]
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age",18,100)
    dependents = st.number_input("Number of Dependents",0,10)
    tenure = st.number_input("Tenure in Months",0,120)

with col2:
    monthly = st.number_input("Monthly Charge",0.0,500.0)
    total = st.number_input("Total Charges",0.0,10000.0)

st.markdown("---")

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

if st.button("🔍 Predict Churn"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][0]

    st.subheader(f"📈 Churn Risk: {probability*100:.1f}%")

    if prediction[0]==0:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

# ===== FOOTER =====
st.markdown("---")
st.caption("Built by Vishal Verma | Machine Learning Project")

# ===== SOCIAL LINKS =====
st.markdown(
    """
    🔗 **Connect with me:**

    [GitHub](https://github.com/VishalVerma0310)  
    [LinkedIn](https://www.linkedin.com/in/vishalverma0310)
    """
)
