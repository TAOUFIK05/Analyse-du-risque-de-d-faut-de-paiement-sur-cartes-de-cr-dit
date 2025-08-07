import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle
model = joblib.load("xgboost_credit_model.pkl")

st.title("💳 Prédiction du Risque de Défaut de Paiement")

st.markdown("Remplissez les informations ci-dessous pour évaluer le risque de crédit :")

# Organiser les entrées en colonnes
col1, col2, col3 = st.columns(3)

with col1:
    limit_bal = st.number_input("💰 Crédit autorisé (LIMIT_BAL)", min_value=0, max_value=1000000, step=10000)
    age = st.number_input("🎂 Âge", min_value=18, max_value=100)
    sex = st.selectbox("Sexe", [1, 2], format_func=lambda x: "Homme" if x == 1 else "Femme")
    education = st.selectbox("🎓 Éducation", [1, 2, 3, 4], format_func=lambda x: {1: "Université", 2: "Supérieur", 3: "Lycée", 4: "Autre"}[x])
    marriage = st.selectbox("💍 Statut marital", [1, 2, 3], format_func=lambda x: {1: "Marié", 2: "Célibataire", 3: "Autre"}[x])

with col2:
    pay_0 = st.number_input("PAY_0 (dernier mois)", step=1)
    pay_2 = st.number_input("PAY_2", step=1)
    pay_3 = st.number_input("PAY_3", step=1)
    pay_4 = st.number_input("PAY_4", step=1)
    pay_5 = st.number_input("PAY_5", step=1)
    pay_6 = st.number_input("PAY_6", step=1)

with col3:
    bill_amt1 = st.number_input("BILL_AMT1", step=1000)
    bill_amt2 = st.number_input("BILL_AMT2", step=1000)
    bill_amt3 = st.number_input("BILL_AMT3", step=1000)
    bill_amt4 = st.number_input("BILL_AMT4", step=1000)
    bill_amt5 = st.number_input("BILL_AMT5", step=1000)
    bill_amt6 = st.number_input("BILL_AMT6", step=1000)

# Ligne supplémentaire pour les montants payés
st.markdown("### Paiements effectués :")
col4, col5, col6 = st.columns(3)

with col4:
    pay_amt1 = st.number_input("PAY_AMT1", step=1000)
    pay_amt2 = st.number_input("PAY_AMT2", step=1000)

with col5:
    pay_amt3 = st.number_input("PAY_AMT3", step=1000)
    pay_amt4 = st.number_input("PAY_AMT4", step=1000)

with col6:
    pay_amt5 = st.number_input("PAY_AMT5", step=1000)
    pay_amt6 = st.number_input("PAY_AMT6", step=1000)

# Préparation des données
input_data = pd.DataFrame([{
    'LIMIT_BAL': limit_bal,
    'SEX': sex,
    'EDUCATION': education,
    'MARRIAGE': marriage,
    'AGE': age,
    'PAY_0': pay_0,
    'PAY_2': pay_2,
    'PAY_3': pay_3,
    'PAY_4': pay_4,
    'PAY_5': pay_5,
    'PAY_6': pay_6,
    'BILL_AMT1': bill_amt1,
    'BILL_AMT2': bill_amt2,
    'BILL_AMT3': bill_amt3,
    'BILL_AMT4': bill_amt4,
    'BILL_AMT5': bill_amt5,
    'BILL_AMT6': bill_amt6,
    'PAY_AMT1': pay_amt1,
    'PAY_AMT2': pay_amt2,
    'PAY_AMT3': pay_amt3,
    'PAY_AMT4': pay_amt4,
    'PAY_AMT5': pay_amt5,
    'PAY_AMT6': pay_amt6
}])

# Prédiction
if st.button("🎯 Prédire le risque"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ Risque élevé de défaut ! Probabilité : {probability:.2%}")
    else:
        st.success(f"✅ Faible risque de défaut. Probabilité : {probability:.2%}")
