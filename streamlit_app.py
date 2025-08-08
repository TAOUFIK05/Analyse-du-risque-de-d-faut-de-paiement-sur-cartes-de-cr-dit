import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# === Configuration de la page ===
st.set_page_config(page_title="💳 Risque de Défaut de Paiement", layout="wide")

# === Styles CSS ===
CUSTOM_CSS = """
<style>
/* Bouton prédiction */
div.stButton > button {
    background-color: #0099ff;
    color: white;
    font-size: 18px;
    height: 50px;
    width: 250px;
    border-radius: 10px;
    font-weight: bold;
    margin: auto;
    display: block;
}
div.stButton > button:hover {
    background-color: #007acc;
    color: #fff;
}
/* Fond onglet Prédiction */
.prediction-section {
    background-color: #f0f9ff;
    padding: 20px;
    border-radius: 10px;
}
/* Fond onglet À propos */
.about-section {
    background-color: #fff7e6;
    padding: 20px;
    border-radius: 10px;
}
/* Fond jauge */
.result-section {
    background-color: #f5f5ff;
    padding: 20px;
    border-radius: 10px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# === Charger le modèle ===
model, feature_names = joblib.load("xgboost_credit_model.pkl")

# === Onglets ===
tab1, tab2 = st.tabs(["💡 Prédiction", "ℹ️ À propos"])

# ==== Onglet À propos ====
with tab2:
    st.markdown('<div class="about-section">', unsafe_allow_html=True)
    st.title("À propos de l'application")
    st.markdown("""
    Cette application prédit le **risque de défaut de paiement** sur carte de crédit grâce à un modèle **XGBoost**.

    - **📊 Données** : 30 000 clients avec 23 variables explicatives.
    - **🔑 Variables clés** : Limite de crédit, historique de paiement, montants dus, etc.
    - **📈 Résultat** : Risque faible ou élevé avec probabilité.
    
    💡 Utilisez l'onglet **Prédiction** pour tester avec vos données.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==== Onglet Prédiction ====
with tab1:
    st.markdown('<div class="prediction-section">', unsafe_allow_html=True)
    st.title("💳 Prédiction du Risque de Défaut")

    col1, col2 = st.columns(2)

    with col1:
        limit_bal = st.number_input("💰 Montant crédit (LIMIT_BAL)", min_value=0, max_value=1_000_000, step=10_000, value=500_000)
        sex = st.selectbox("👤 Sexe", options=[1, 2], format_func=lambda x: {1: "Homme", 2: "Femme"}[x])
        education = st.selectbox("🎓 Éducation", options=[1, 2, 3, 4], format_func=lambda x: {1: "Université", 2: "École Supérieure", 3: "Lycée", 4: "Autre"}[x])
        marriage = st.selectbox("💍 Statut marital", options=[1, 2, 3], format_func=lambda x: {1: "Marié", 2: "Célibataire", 3: "Autre"}[x])
        age = st.number_input("🎂 Âge", min_value=18, max_value=100, step=1, value=35)
        pay_0 = st.number_input("Paiement PAY_0 (dernier mois)", min_value=-2, max_value=9, step=1, value=0)
        pay_2 = st.number_input("Paiement PAY_2 (-2 mois)", min_value=-2, max_value=9, step=1, value=0)
        pay_3 = st.number_input("Paiement PAY_3 (-3 mois)", min_value=-2, max_value=9, step=1, value=0)

    with col2:
        pay_4 = st.number_input("Paiement PAY_4 (-4 mois)", min_value=-2, max_value=9, step=1, value=0)
        pay_5 = st.number_input("Paiement PAY_5 (-5 mois)", min_value=-2, max_value=9, step=1, value=0)
        pay_6 = st.number_input("Paiement PAY_6 (-6 mois)", min_value=-2, max_value=9, step=1, value=0)
        bill_amt1 = st.number_input("Facture BILL_AMT1 (dernier mois)", step=1000, value=0)
        bill_amt2 = st.number_input("Facture BILL_AMT2 (-1 mois)", step=1000, value=0)
        bill_amt3 = st.number_input("Facture BILL_AMT3 (-2 mois)", step=1000, value=0)
        bill_amt4 = st.number_input("Facture BILL_AMT4 (-3 mois)", step=1000, value=0)
        bill_amt5 = st.number_input("Facture BILL_AMT5 (-4 mois)", step=1000, value=0)
        bill_amt6 = st.number_input("Facture BILL_AMT6 (-5 mois)", step=1000, value=0)

    pay_amt1 = st.number_input("Paiement PAY_AMT1 (dernier mois)", step=1000, value=0)
    pay_amt2 = st.number_input("Paiement PAY_AMT2 (-1 mois)", step=1000, value=0)
    pay_amt3 = st.number_input("Paiement PAY_AMT3 (-2 mois)", step=1000, value=0)
    pay_amt4 = st.number_input("Paiement PAY_AMT4 (-3 mois)", step=1000, value=0)
    pay_amt5 = st.number_input("Paiement PAY_AMT5 (-4 mois)", step=1000, value=0)
    pay_amt6 = st.number_input("Paiement PAY_AMT6 (-5 mois)", step=1000, value=0)

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

    user_data = user_data[feature_names]

    if st.button("🔍 Calculer le risque"):
        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0][1]

        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilité de défaut (%)", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))

        st.plotly_chart(fig, use_container_width=True)

        if prediction == 1:
            st.error(f"⚠️ Risque élevé : {probability:.2%}")
            st.info("💡 Conseil : Réduire les dettes et payer à temps.")
        else:
            st.success(f"✅ Faible risque : {probability:.2%}")
            st.info("👍 Continuez votre bonne gestion financière.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
