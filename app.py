import streamlit as st
import numpy as np
import pickle

import shap

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="centered"
)

# ===== GLASSMORPHISM CSS =====
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
background: linear-gradient(135deg,#0f172a,#1e293b);
color:white;
}

.glass {
background: rgba(255,255,255,0.08);
padding:20px;
border-radius:15px;
backdrop-filter: blur(10px);
box-shadow: 0 4px 30px rgba(0,0,0,0.3);
}

footer {visibility:hidden;}

.premium-footer {
position:fixed;
bottom:0;
width:100%;
background:#020617;
color:white;
text-align:center;
padding:10px;
}

.premium-footer img{
filter:invert(1);
margin:0 10px;
}

</style>
""", unsafe_allow_html=True)

# ===== LOAD MODEL =====
model = pickle.load(open("model.pkl","rb"))

# ===== HEADER =====
st.markdown("## 📊 Customer Churn Predictor")
st.markdown("AI-powered telecom churn prediction")

# ===== SAMPLE DATA BUTTON =====
if st.button("✨ Autofill Sample Data"):
    st.session_state.sample = True

sample = st.session_state.get("sample", False)

# ===== INPUT UI =====
st.markdown('<div class="glass">', unsafe_allow_html=True)

gender = st.selectbox("Gender", ["Male","Female"], index=0 if sample else 0)
married = st.selectbox("Married", ["Yes","No"], index=0 if sample else 0)
internet = st.selectbox("Internet Service", ["Yes","No"], index=0 if sample else 0)

contract = st.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"],
    index=0 if sample else 0
)

payment = st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer","Credit card"],
    index=0 if sample else 0
)

col1,col2 = st.columns(2)

with col1:
    age = st.number_input("Age",18,100, 45 if sample else 18)
    dependents = st.number_input("Dependents",0,10, 2 if sample else 0)
    tenure = st.number_input("Tenure",0,120, 36 if sample else 0)

with col2:
    monthly = st.number_input("Monthly Charges",0.0,500.0, 70.0 if sample else 0.0)
    total = st.number_input("Total Charges",0.0,10000.0, 2500.0 if sample else 0.0)

st.markdown('</div>', unsafe_allow_html=True)

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

    # Animated bar
    st.progress(float(prob))

    st.metric("Churn Probability", f"{prob*100:.1f}%")

    if pred[0]==0:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Customer Likely to Stay")

    # ===== SHAP EXPLAINABILITY =====
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        st.subheader("🔍 Prediction Explanation")
        shap.initjs()
        st.write("Feature impact on prediction:")
        st.bar_chart(shap_values[0])
    except:
        st.info("Explainability not available for this model type.")

# ===== FOOTER =====
st.markdown("""
<div class="premium-footer">

Built by <b>Vishal Verma</b>

<br>

<a href="https://github.com/VishalVerma0310" target="_blank">
<img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/github.svg" width="20">
</a>

<a href="https://www.linkedin.com/in/vishalverma0310" target="_blank">
<img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/linkedin.svg" width="20">
</a>

</div>
""", unsafe_allow_html=True)
