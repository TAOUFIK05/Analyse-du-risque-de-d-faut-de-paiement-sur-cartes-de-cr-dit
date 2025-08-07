import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger le modèle
model = joblib.load("xgboost_credit_model.pkl")

st.title("💳 Prédiction du Risque de Défaut de Paiement (XGBoost)")
st.markdown("Remplissez les informations ci-dessous pour évaluer le risque de crédit.")

# === Entrée utilisateur ===
limit_bal = st.number_input("Montant de crédit autorisé (LIMIT_BAL)", min_value=0, max_value=1000000, step=10000)
age = st.number_input("Âge du client", min_value=18, max_value=100, step=1)
education = st.selectbox("Niveau d'éducation", options=[1, 2, 3, 4],
                         format_func=lambda x: {1: "Université", 2: "École Supérieure", 3: "Lycée", 4: "Autre"}[x])
marriage = st.selectbox("Statut marital", options=[1, 2, 3],
                        format_func=lambda x: {1: "Marié", 2: "Célibataire", 3: "Autre"}[x])
sex = st.selectbox("Sexe", options=[1, 2],
                   format_func=lambda x: {1: "Homme", 2: "Femme"}[x])
bill_amt1 = st.number_input("Montant de la facture du mois précédent (BILL_AMT1)", step=1000)
pay_amt1 = st.number_input("Montant payé le mois précédent (PAY_AMT1)", step=1000)

# Préparation des données
input_data = pd.DataFrame([{
    'LIMIT_BAL': limit_bal,
    'AGE': age,
    'EDUCATION': education,
    'MARRIAGE': marriage,
    'SEX': sex,
    'BILL_AMT1': bill_amt1,
    'PAY_AMT1': pay_amt1
}])

# Prédiction
if st.button("📊 Prédire"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Risque élevé de défaut. Probabilité : {probability:.2%}")
    else:
        st.success(f"✅ Faible risque de défaut. Probabilité : {probability:.2%}")
