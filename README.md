# 💳 Analyse du Risque de Défaut de Paiement sur Cartes de Crédit

## 🔗 Accès à l'application Streamlit

👉 Tester l'application [ici](https://risque-creditt.streamlit.app/)
<a href="https://risque-creditt.streamlit.app/" target="_blank">Tester l'application</a>
---

## 📁 À propos du dataset

Ce projet repose sur le dataset public **"Default of Credit Card Clients"**, proposé par l’[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).  
Il contient les informations de **30 000 clients** de cartes de crédit à Taïwan, enregistrées entre avril et septembre 2005.

Le jeu de données inclut :
- Des données **démographiques** (âge, sexe, statut marital, éducation…)
- L’**historique de paiement** sur 6 mois (`PAY_0`, ..., `PAY_6`)
- Les **montants facturés** (`BILL_AMT1` à `BILL_AMT6`)
- Les **paiements effectués** (`PAY_AMT1` à `PAY_AMT6`)
- Une variable cible : `default_payment_next_month` (1 = défaut, 0 = non-défaut)

📚 **Référence** :  
Lichman, M. (2013). *UCI Machine Learning Repository*. University of California, Irvine.

---

## ❓ Problématique

Les établissements financiers doivent pouvoir **évaluer le risque de crédit** de leurs clients pour limiter les défauts de paiement.

> **Question centrale** :  
> _Peut-on prédire le défaut de paiement d’un client le mois suivant, à partir de ses données démographiques et historiques financiers ?_

---

## 🎯 Objectifs du projet

Développer un modèle de **Machine Learning** capable de :

✅ Prédire la probabilité qu’un client fasse défaut  
✅ Identifier les variables les plus influentes  
✅ Aider à la prise de décision pour l’octroi de crédit  

---

## 🧪 Étapes du projet

1. **📂 Compréhension du dataset**  
   Description, types de variables, données manquantes  

2. **📊 Analyse exploratoire (EDA)**  
   Visualisations, corrélations, insights  

3. **🧹 Prétraitement des données**  
   Nettoyage, encodage, normalisation  

4. **🤖 Modélisation**  
   Modèles testés :  
   - Régression logistique  
   - Arbre de décision  
   - Random Forest  
   - **XGBoost** (retenu comme modèle final)

5. **📈 Évaluation des performances**  
   - Accuracy  
   - Recall  
   - Precision  
   - AUC-ROC  

6. **🔍 Interprétation & Visualisation**  
   - Importance des variables  
   - SHAP (interprétabilité)

---

## 🖥️ Interface Web avec Streamlit

Une application Streamlit interactive a été développée pour :

- ✅ Saisir les données d’un client
- ✅ Obtenir une **prédiction instantanée**
- ✅ Visualiser la probabilité de défaut dans un graphique clair

📍 Lien vers l’application :  
👉 [https://risque-creditt.streamlit.app/](https://risque-creditt.streamlit.app/)

---

## 🚀 Lancer l’application en local

### 1. Cloner le dépôt

```bash
git clone https://github.com/TAOUFIK05/analyse-du-risque-de-defaut-de-paiement
cd analyse-du-risque-de-defaut-de-paiement
