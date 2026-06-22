import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = joblib.load('../models/xgboost_model.pkl')
    medians = pd.read_csv('../models/feature_medians.csv',
                           index_col=0, header=None).squeeze()
    return model, medians

model, feature_medians = load_artifacts()

# ── Scoring function ─────────────────────────────────────────────────────────
def probability_to_score(probability, min_score=300, max_score=850):
    probability = np.clip(probability, 0.0001, 0.9999)
    log_odds    = np.log(1 - probability) - np.log(probability)
    min_lo      = np.log(1 - 0.9999) - np.log(0.9999)
    max_lo      = np.log(1 - 0.0001) - np.log(0.0001)
    score       = min_score + (max_score - min_score) * (
                    (log_odds - min_lo) / (max_lo - min_lo))
    return int(np.round(score))

# ── Employment length encoding ───────────────────────────────────────────────
emp_length_map = {
    '< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,
    '5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,
    '10+ years':10,'Unknown':-1
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Credit Risk Scoring Engine", page_icon="🏦")
st.title("🏦 Credit Risk Scoring Engine")
st.markdown("Enter applicant details to generate a credit score and risk explanation.")

# ── Input form ───────────────────────────────────────────────────────────────
st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    loan_amnt  = st.number_input("Loan Amount (₹)",
                    min_value=10000, max_value=5000000,
                    value=500000, step=10000)
    annual_inc = st.number_input("Annual Income (₹)",
                    min_value=100000, max_value=10000000,
                    value=600000, step=10000)
    dti        = st.number_input("Debt-to-Income Ratio (DTI %)",
                    min_value=0.0, max_value=60.0,
                    value=15.0, step=0.5)
    fico       = st.number_input("FICO Score",
                    min_value=580, max_value=850,
                    value=700, step=5)

with col2:
    term       = st.selectbox("Loan Term (months)", options=[36, 60])
    emp_length = st.selectbox("Employment Length", options=[
                    '< 1 year','1 year','2 years','3 years','4 years',
                    '5 years','6 years','7 years','8 years','9 years',
                    '10+ years','Unknown'])
    purpose    = st.selectbox("Loan Purpose", options=[
                    'debt_consolidation','credit_card','home_improvement',
                    'other','major_purchase','medical','small_business',
                    'car','moving','vacation','house','wedding',
                    'renewable_energy','educational'])
    home_ownership = st.selectbox("Home Ownership", options=[
                    'RENT','MORTGAGE','OWN','OTHER'])

# ── Score button ─────────────────────────────────────────────────────────────
if st.button("Generate Credit Score", type="primary"):

    # Start with median values for all features
    input_data = feature_medians.copy()

    # Override with user inputs
    input_data['loan_amnt']      = loan_amnt
    input_data['annual_inc']     = annual_inc
    input_data['dti']            = dti
    input_data['fico_range_low'] = fico
    input_data['term']           = term
    input_data['emp_length']     = emp_length_map[emp_length]

    # Fix lender-assigned features at median
    input_data['sub_grade'] = feature_medians['sub_grade']
    input_data['int_rate']  = feature_medians['int_rate']

    # Handle purpose one-hot encoding
    purpose_cols = [c for c in model.feature_names_in_ if c.startswith('purpose_')]
    for col in purpose_cols:
        input_data[col] = 0
    purpose_col = f'purpose_{purpose}'
    if purpose_col in input_data.index:
        input_data[purpose_col] = 1

    # Handle home_ownership one-hot encoding
    ownership_cols = [c for c in model.feature_names_in_ if c.startswith('home_ownership_')]
    for col in ownership_cols:
        input_data[col] = 0
    ownership_col = f'home_ownership_{home_ownership}'
    if ownership_col in input_data.index:
        input_data[ownership_col] = 1

    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)

    # Predict
    prob  = model.predict_proba(input_df)[0][1]
    score = probability_to_score(prob)

    # ── Score display ─────────────────────────────────────────────────────
    st.divider()
    st.header("Credit Score Result")

    if score >= 590:
        color  = "green"
        rating = "Good"
    elif score >= 550:
        color  = "orange"
        rating = "Fair"
    else:
        color  = "red"
        rating = "Poor"

    st.markdown(f"## <span style='color:{color}'>{score} — {rating}</span>",
                unsafe_allow_html=True)
    st.progress((score - 300) / 550)
    st.caption(f"Estimated default probability: {prob*100:.1f}%")

    # ── Risk factors ──────────────────────────────────────────────────────
    st.subheader("Key Risk Factors")

    reasons = []

    if dti > 30:
        reasons.append("🔴 High debt-to-income ratio increases risk")
    elif dti < 10:
        reasons.append("🟢 Low debt-to-income ratio reduces risk")

    if fico >= 720:
        reasons.append("🟢 Strong FICO score reduces risk")
    elif fico < 650:
        reasons.append("🔴 Low FICO score increases risk")

    if emp_length_map[emp_length] >= 7:
        reasons.append("🟢 Long employment history reduces risk")
    elif emp_length_map[emp_length] <= 0:
        reasons.append("🔴 Short or unknown employment history increases risk")

    if loan_amnt > 2000000:
        reasons.append("🔴 Large loan amount increases risk")
    else:
        reasons.append("🟢 Loan amount is within a manageable range")

    if term == 60:
        reasons.append("🔴 60-month loans carry higher default risk than 36-month")
    else:
        reasons.append("🟢 36-month term carries lower default risk")

    for r in reasons[:3]:
        st.markdown(r)

    st.divider()
    st.caption("Model: XGBoost | Data: LendingClub 2007–2018 | AUC: 0.730")