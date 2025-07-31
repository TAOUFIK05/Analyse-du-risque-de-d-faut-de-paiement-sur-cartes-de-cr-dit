# Analyse du risque de défaut de paiement sur cartes de crédit

## À propos du dataset

Ce projet est basé sur le dataset public **"Default of Credit Card Clients"**, publié par l’[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

Le jeu de données contient des informations sur les paiements par carte de crédit de 30 000 clients à Taïwan, enregistrées entre avril et septembre 2005.Il comporte 25 variables décrivant les caractéristiques démographiques, les historiques de paiements, les montants facturés et remboursés, ainsi qu’une variable cible indiquant si le client a fait défaut le mois suivant.

### Source officielle :
UCI Machine Learning Repository – [https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

### Remerciements :
 > Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

## Problématique

Les établissements de crédit cherchent constamment à évaluer le risque que représente chaque client. Identifier à l'avance les clients susceptibles de faire défaut permet de limiter les pertes financières.

**Problème posé** :  
Peut-on prédire si un client fera défaut de paiement le mois suivant en se basant sur ses caractéristiques démographiques, son historique de paiements et ses habitudes financières ?

L’objectif de ce projet est donc de construire un modèle de machine learning capable de prédire le défaut de paiement, afin d'aider à la prise de décision en matière d'octroi de crédit.

## Objectifs et étapes du projet

Voici les grandes étapes que vous retrouverez dans le notebook :

1. **Présentation et compréhension du dataset**
   - Description des variables
   - Analyse des valeurs manquantes et des types

2. **Analyse exploratoire des données (EDA)**
   - Étude des distributions
   - Visualisation des relations entre variables
   - Corrélations avec la variable cible

3. **Préparation des données**
   - Nettoyage
   - Encodage des variables catégorielles
   - Normalisation / standardisation

4. **Modélisation**
   - Entraînement de plusieurs modèles (régression logistique, arbre de décision, random forest, etc.)
   - Évaluation des performances (accuracy, précision, rappel, AUC...)

5. **Interprétation des résultats**
   - Importance des variables
   - Explicabilité avec SHAP (si applicable)

6. **Conclusion**
   - Résumé des résultats
   - Limites et pistes d'amélioration
