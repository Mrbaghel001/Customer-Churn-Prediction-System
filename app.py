import streamlit as st
import pickle
import pandas as pd

# Load model and columns
model = pickle.load(open("model/churn_model.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("Fill customer details to predict churn behavior")

# ================= INPUT SECTION =================

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (Months)", 1, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", value=70.0)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

# ================= PREDICTION =================

if st.button("🔍 Predict Churn"):

    # Base input
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
    }

    df_input = pd.DataFrame([input_data])

    # ================= HANDLE CATEGORICAL =================

    # Contract
    df_input['Contract_One year'] = 1 if contract == "One year" else 0
    df_input['Contract_Two year'] = 1 if contract == "Two year" else 0

    # Internet Service
    df_input['InternetService_Fiber optic'] = 1 if internet_service == "Fiber optic" else 0
    df_input['InternetService_No'] = 1 if internet_service == "No" else 0

    # Gender
    df_input['gender_Male'] = 1 if gender == "Male" else 0

    # Dependents
    df_input['Dependents_Yes'] = 1 if dependents == "Yes" else 0

    # ================= MATCH TRAINING COLUMNS =================

    for col in columns:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[columns]

    # ================= PREDICT =================

    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    # ================= OUTPUT =================

    st.subheader("📈 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Customer will churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer will stay (Probability: {1 - probability:.2f})")