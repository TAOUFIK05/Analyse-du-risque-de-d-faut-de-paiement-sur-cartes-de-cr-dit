import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Charger le modèle et la liste des features ===
model, feature_names = joblib.load("xgboost_credit_model.pkl")

st.title("Prédiction du Risque de Défaut de Paiement")
st.markdown("Remplissez les informations ci-dessous pour évaluer le risque de crédit.")

# === Interface utilisateur ===
limit_bal = st.number_input("💰 Montant de crédit autorisé (LIMIT_BAL)", min_value=0, max_value=1000000, step=10000)
sex = st.selectbox("Sexe (SEX)", options=[1, 2], format_func=lambda x: {1: "Homme", 2: "Femme"}.get(x))
education = st.selectbox("🎓 Niveau d'éducation (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: {1: "Université", 2: "École Supérieure", 3: "Lycée", 4: "Autre"}.get(x))
marriage = st.selectbox("💍 Statut marital (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: {1: "Marié", 2: "Célibataire", 3: "Autre"}.get(x))
age = st.number_input("🎂 Âge du client (AGE)", min_value=18, max_value=100, step=1)

# Historique des paiements (statuts)
pay_0 = st.number_input("📅 Statut de remboursement dernier mois (PAY_0)", min_value=-2, max_value=9, step=1)
pay_2 = st.number_input("📅 Statut de remboursement il y a 2 mois (PAY_2)", min_value=-2, max_value=9, step=1)
pay_3 = st.number_input("📅 Statut de remboursement il y a 3 mois (PAY_3)", min_value=-2, max_value=9, step=1)
pay_4 = st.number_input("📅 Statut de remboursement il y a 4 mois (PAY_4)", min_value=-2, max_value=9, step=1)
pay_5 = st.number_input("📅 Statut de remboursement il y a 5 mois (PAY_5)", min_value=-2, max_value=9, step=1)
pay_6 = st.number_input("📅 Statut de remboursement il y a 6 mois (PAY_6)", min_value=-2, max_value=9, step=1)

# Montants des factures (BILL_AMT)
bill_amt1 = st.number_input("💳 Montant facture mois-1 (BILL_AMT1)", step=1000)
bill_amt2 = st.number_input("💳 Montant facture mois-2 (BILL_AMT2)", step=1000)
bill_amt3 = st.number_input("💳 Montant facture mois-3 (BILL_AMT3)", step=1000)
bill_amt4 = st.number_input("💳 Montant facture mois-4 (BILL_AMT4)", step=1000)
bill_amt5 = st.number_input("💳 Montant facture mois-5 (BILL_AMT5)", step=1000)
bill_amt6 = st.number_input("💳 Montant facture mois-6 (BILL_AMT6)", step=1000)

# Montants remboursés (PAY_AMT)
pay_amt1 = st.number_input("💵 Montant payé mois-1 (PAY_AMT1)", step=1000)
pay_amt2 = st.number_input("💵 Montant payé mois-2 (PAY_AMT2)", step=1000)
pay_amt3 = st.number_input("💵 Montant payé mois-3 (PAY_AMT3)", step=1000)
pay_amt4 = st.number_input("💵 Montant payé mois-4 (PAY_AMT4)", step=1000)
pay_amt5 = st.number_input("💵 Montant payé mois-5 (PAY_AMT5)", step=1000)
pay_amt6 = st.number_input("💵 Montant payé mois-6 (PAY_AMT6)", step=1000)

# === Rassembler les données dans un DataFrame ===
user_data = pd.DataFrame([{
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

# Réorganiser les colonnes pour correspondre au modèle
user_data = user_data[feature_names]

# 💅 Style du bouton
st.markdown("""
    <style>
        div.stButton > button {
            font-size: 20px;
            padding: 15px 30px;
            border-radius: 10px;
            background-color: #4CAF50;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# === Prédiction ===
if st.button("🎯 Prédire"):
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ Risque ÉLEVÉ de défaut.\n\nProbabilité : **{probability:.2%}**")
    else:
        st.success(f"✅ Faible risque de défaut.\n\nProbabilité : **{probability:.2%}**")
        


