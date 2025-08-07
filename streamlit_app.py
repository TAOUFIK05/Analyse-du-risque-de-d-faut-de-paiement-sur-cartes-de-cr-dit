import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# === Charger le modèle et les noms de variables ===
model, feature_names = joblib.load("xgboost_credit_model.pkl")

# === Titre et instructions ===
st.set_page_config(page_title="Risque de Défaut de Paiement", layout="wide")
st.title("💳 Prédiction du Risque de Défaut de Paiement (XGBoost)")
st.markdown("Remplissez les informations ci-dessous pour évaluer le risque de crédit d'un client.")

# === Mise en page : deux colonnes ===
col1, col2 = st.columns(2)

with col1:
    limit_bal = st.number_input("💰 Crédit autorisé (LIMIT_BAL)", min_value=0, max_value=1000000, step=10000)
    sex = st.selectbox("🧍 Sexe", options=[1, 2], format_func=lambda x: {1: "Homme", 2: "Femme"}.get(x))
    education = st.selectbox("🎓 Niveau d'éducation", options=[1, 2, 3, 4],
                             format_func=lambda x: {1: "Université", 2: "École Supérieure", 3: "Lycée", 4: "Autre"}.get(x))
    marriage = st.selectbox("💍 Statut marital", options=[1, 2, 3],
                            format_func=lambda x: {1: "Marié", 2: "Célibataire", 3: "Autre"}.get(x))
    age = st.number_input("🎂 Âge", min_value=18, max_value=100, step=1)

with col2:
    pay_0 = st.number_input("📆 Retard paiement septembre (PAY_0)", min_value=-2, max_value=9, step=1)
    pay_2 = st.number_input("📆 Retard paiement août (PAY_2)", min_value=-2, max_value=9, step=1)
    pay_3 = st.number_input("📆 Retard paiement juillet (PAY_3)", min_value=-2, max_value=9, step=1)
    pay_4 = st.number_input("📆 Retard paiement juin (PAY_4)", min_value=-2, max_value=9, step=1)
    pay_5 = st.number_input("📆 Retard paiement mai (PAY_5)", min_value=-2, max_value=9, step=1)
    pay_6 = st.number_input("📆 Retard paiement avril (PAY_6)", min_value=-2, max_value=9, step=1)

# === Section secondaire : montants des factures et paiements ===
with st.expander("💸 Détails des montants de factures et paiements (cliquer pour développer)", expanded=False):
    bill_amt1 = st.number_input("📄 Montant facture septembre (BILL_AMT1)", step=1000)
    bill_amt2 = st.number_input("📄 Montant facture août (BILL_AMT2)", step=1000)
    bill_amt3 = st.number_input("📄 Montant facture juillet (BILL_AMT3)", step=1000)
    bill_amt4 = st.number_input("📄 Montant facture juin (BILL_AMT4)", step=1000)
    bill_amt5 = st.number_input("📄 Montant facture mai (BILL_AMT5)", step=1000)
    bill_amt6 = st.number_input("📄 Montant facture avril (BILL_AMT6)", step=1000)

    pay_amt1 = st.number_input("💵 Paiement effectué septembre (PAY_AMT1)", step=1000)
    pay_amt2 = st.number_input("💵 Paiement effectué août (PAY_AMT2)", step=1000)
    pay_amt3 = st.number_input("💵 Paiement effectué juillet (PAY_AMT3)", step=1000)
    pay_amt4 = st.number_input("💵 Paiement effectué juin (PAY_AMT4)", step=1000)
    pay_amt5 = st.number_input("💵 Paiement effectué mai (PAY_AMT5)", step=1000)
    pay_amt6 = st.number_input("💵 Paiement effectué avril (PAY_AMT6)", step=1000)

# === Rassembler toutes les données utilisateur ===
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

# Réorganiser les colonnes
user_data = user_data[feature_names]

# === Prédiction ===
predict_btn = st.button("🔍 Évaluer le risque", use_container_width=True)

if predict_btn:
    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1]  # proba d'être en défaut

    if prediction == 1:
        st.error(f"🚨 **Risque ÉLEVÉ** de défaut de paiement.")
    else:
        st.success(f"✅ **Faible risque** de défaut de paiement.")

    st.markdown("### 🔢 Probabilité de défaut :")
    # Jauge avec Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Probabilité (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red" if prediction == 1 else "green"},
            'steps': [
                {'range': [0, 50], 'color': "#DFF0D8"},
                {'range': [50, 100], 'color': "#F2DEDE"}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

