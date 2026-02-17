import streamlit as st
import numpy as np
import pickle

# ===== PAGE SETTINGS =====
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# ===== PREMIUM CSS =====
st.markdown("""
<style>

footer {visibility: hidden;}

.premium-footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background: linear-gradient(90deg,#0f172a,#111827);
    color: white;
    text-align: center;
    padding: 12px 0;
    font-size: 14px;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.4);
    z-index: 999;
}

.premium-footer a {
    margin: 0 12px;
    display: inline-block;
    transition: transform 0.3s ease, filter 0.3s ease;
}

.premium-footer a:hover {
    transform: scale(1.2);
    filter: drop-shadow(0 0 6px #38bdf8);
}

.premium-footer img {
    filter: invert(1);
}

</style>
""", unsafe_allow_html=True)

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

# ===== PREMIUM FOOTER =====

st.markdown("""
<div class="premium-footer">

Built by <b>Vishal Verma</b> | Machine Learning Project

<br>

<a href="https://github.com/VishalVerma0310" target="_blank">
<img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg" width="22">
</a>

<a href="https://www.linkedin.com/in/YOUR-LINKEDIN-ID" target="_blank">
<img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/linkedin.svg" width="22">
</a>

</div>
""", unsafe_allow_html=True)
