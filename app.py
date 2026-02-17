import streamlit as st
import numpy as np
import pickle

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📈",
    layout="centered"
)

# ===== CSS =====
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

/* Hide default footer */
footer {visibility:hidden;}

/* Footer */
.premium-footer{
position:fixed;
bottom:0;
left:0;
width:100%;
background:#020617;
color:white;
text-align:center;
padding:14px 0;
font-size:14px;
z-index:999;
}

/* Icons */
.premium-footer img{
filter:invert(1);
margin:0 12px;
transition: all 0.3s ease;
}

/* Glow hover */
.premium-footer img:hover{
transform:scale(1.25);
filter:invert(1) drop-shadow(0 0 8px #38bdf8);
}

</style>
""", unsafe_allow_html=True)

# ===== LOAD MODEL =====
model = pickle.load(open("model.pkl","rb"))

# ===== HEADER =====
st.markdown("## 📊 Customer Churn Predictor")
st.markdown("AI-powered telecom churn prediction")
st.markdown("---")

# ===== INPUTS =====

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

col1,col2 = st.columns(2)

with col1:
    age = st.number_input("Age",18,100)
    dependents = st.number_input("Dependents",0,10)
    tenure = st.number_input("Tenure (Months)",0,120)

with col2:
    monthly = st.number_input("Monthly Charges",0.0,500.0)
    total = st.number_input("Total Charges",0.0,10000.0)

st.markdown("---")

# ===== ENCODING =====
gender = 1 if gender=="Male" else 0
married = 1 if married=="Yes" else 0
internet = 1 if internet=="Yes" else 0

contract = {"Month-to-month":0,"One year":1,"Two year":2}[contract]

payment = {
"Electronic check":0,
"Mailed check":1,
"Bank transfer":2,
"Credit card":3
}[payment]

input_data = np.array([[gender,age,married,
dependents,tenure,
internet,contract,
payment,monthly,total]])

# ===== PREDICTION =====
if st.button("🚀 Predict"):

    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][0]

    st.subheader("📈 Churn Risk Meter")
    st.progress(float(prob))

    st.metric("Churn Probability", f"{prob*100:.1f}%")

    if pred[0]==0:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Customer Likely to Stay")

# ===== FOOTER =====
st.markdown("""
<div class="premium-footer">

Built by <b>Vishal Verma</b>

<br>

<a href="https://github.com/VishalVerma0310" target="_blank">
<img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg" width="22">
</a>

<a href="https://www.linkedin.com/in/vishalverma0310" target="_blank">
<img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/linkedin.svg" width="22">
</a>

</div>
""", unsafe_allow_html=True)
