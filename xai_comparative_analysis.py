#!/usr/bin/env python
# coding: utf-8

# ## SHAPLEY Values

# In[1]:


get_ipython().system('pip install xgboost shap')


# In[2]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import shap


# In[3]:


# Créons un dataset synthétique sur l'approbation de prêts bancaires
np.random.seed(42)
n_samples = 1000

# Générons des features
data = {
    'age': np.random.normal(35, 10, n_samples),
    'salaire_annuel': np.random.normal(45000, 15000, n_samples),
    'anciennete_emploi': np.random.normal(5, 3, n_samples),
    'ratio_dette': np.random.normal(0.3, 0.1, n_samples),
    'historique_credit': np.random.normal(700, 50, n_samples)
}

df = pd.DataFrame(data)


# In[4]:


# Créons la variable target 
def create_target(row):
    score = 0
    score += 1 if row['age'] > 25 else 0
    score += 2 if row['salaire_annuel'] > 40000 else 0
    score += 1 if row['anciennete_emploi'] > 2 else 0
    score += 2 if row['ratio_dette'] < 0.4 else 0
    score += 2 if row['historique_credit'] > 650 else 0
    return 1 if score >= 6 else 0

df['approbation'] = df.apply(create_target, axis=1)


# In[5]:


# Séparation features et target
X = df.drop('approbation', axis=1)
y = df['approbation']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[6]:


# Entraînement du modèle XGBoost
model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)


# In[7]:


# Créons l'explainer SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Affichage des résultats
print("Performance du modèle:")
print(metrics.classification_report(y_test, y_pred))

# Visualisation SHAP
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X)
plt.title("Impact des features sur la décision d'approbation de prêt")
plt.tight_layout()
plt.show()


# In[9]:


# Explication pour un cas spécifique
i = 4  # Premier exemple du dataset
print("\nExplication pour un cas spécifique:")
print("\nCaractéristiques du demandeur:")
for col in X.columns:
    print(f"{col}: {X.iloc[i][col]:.2f}")

print("\nDécision du modèle:", "Approuvé" if model.predict(X.iloc[i:i+1])[0] == 1 else "Refusé")

# Affichage de l'explication SHAP pour ce cas
plt.figure(figsize=(10, 6))
shap.force_plot(explainer.expected_value, shap_values[i,:], X.iloc[i,:], matplotlib=True)
plt.title("Explication de la décision pour ce cas spécifique")
plt.tight_layout()
plt.show()


# ## Counterfactuals pour données tabulaires

# In[10]:


get_ipython().system('pip install dice-ml scikit-learn pandas numpy matplotlib seaborn')


# In[22]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import dice_ml


# In[23]:


# Définissons la seed pour reproductibilité
np.random.seed(42)
n_samples = 1000

# Définition des plages de valeurs réalistes
def clip_score(scores, min_val, max_val):
    return np.clip(scores, min_val, max_val)

# Générons des données avec contraintes
data = {
    # GRE score : plage réelle 260-340
    'gre_score': clip_score(
        np.random.normal(310, 15, n_samples),  # Réduction de l'écart-type
        260,
        340
    ),
    
    # GPA : plage réelle 0.0-4.0
    'gpa': clip_score(
        np.random.normal(3.5, 0.3, n_samples),  # Réduction de l'écart-type
        2.0,  # GPA minimum plus réaliste
        4.0
    ),
    
    # Variables catégorielles avec distributions plus réalistes
    'research_exp': np.random.choice(
        ['yes', 'no'],
        n_samples,
        p=[0.3, 0.7]  # 30% ont de l'expérience en recherche
    ),
    
    'university_rating': np.random.choice(
        ['top', 'medium', 'low'],
        n_samples,
        p=[0.2, 0.5, 0.3]  # Distribution plus réaliste
    ),
    
    'department': np.random.choice(
        ['cs', 'engineering', 'business'],
        n_samples,
        p=[0.3, 0.4, 0.3]  # Distribution équilibrée par département
    ),
    
    'recommendation_strength': np.random.choice(
        ['strong', 'medium', 'weak'],
        n_samples,
        p=[0.3, 0.5, 0.2]  # Distribution typique des recommandations
    )
}

# Création du DataFrame
df = pd.DataFrame(data)

# Ajoutons quelques statistiques descriptives
print("\nStatistiques descriptives des variables numériques:")
print(df[['gre_score', 'gpa']].describe())

# Distribution des variables catégorielles
print("\nDistribution des variables catégorielles:")
for col in ['research_exp', 'university_rating', 'department', 'recommendation_strength']:
    print(f"\n{col}:")
    print(df[col].value_counts(normalize=True).round(3) * 100, "%")

# Visualisons les premières lignes
print("\nPremières lignes du dataset:")
df.head()


# In[24]:


# Créons la variable target
def create_target(row):
    score = 0
    score += (row['gre_score'] - 280) / 40  # Max 4 points
    score += row['gpa'] * 2  # Max 8 points
    score += 2 if row['research_exp'] == 'yes' else 0
    score += 2 if row['university_rating'] == 'top' else 1 if row['university_rating'] == 'medium' else 0
    score += 2 if row['recommendation_strength'] == 'strong' else 1 if row['recommendation_strength'] == 'medium' else 0
    return 1 if score > 12 else 0  # Admission si score > 12

df['admission'] = df.apply(create_target, axis=1)

# Visualisons la distribution des admissions
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='admission')
plt.title('Distribution des Admissions')
plt.show()


# In[25]:


# Séparons les features/target
features = df.drop('admission', axis=1)
target = df['admission']

# Variables catégorielles et numériques
categoric_features = ['research_exp', 'university_rating', 'department', 'recommendation_strength']
numeric_features = ['gre_score', 'gpa']

# Préparons les catégories pour OneHotEncoder
cat_types = []
for cat in categoric_features:
    cat_i = list(features[cat].unique())
    cat_types.append(cat_i)

# Pipeline de prétraitement
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(categories=cat_types))])
preprocessor = ColumnTransformer(transformers=[('categorical', categorical_transformer, categoric_features)])


# In[26]:


# Modèle Random Forest
rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)

# Pipeline complet
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf)
])

# Split et entraînement
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
model.fit(X_train, y_train)

# Évaluons le modèle
y_pred = model.predict(X_test)
print("Performance du modèle:")
print(metrics.classification_report(y_test, y_pred))


# In[27]:


# DiCE setup
data_dice = pd.concat([features, target], axis=1)
d = dice_ml.Data(dataframe=data_dice, continuous_features=numeric_features, outcome_name='admission')
m = dice_ml.Model(model=model, backend='sklearn')
exp = dice_ml.Dice(d, m, method="random")


# In[28]:


# Sélection d'un étudiant
student_index = 10

# Affichage du profil
print("Profil de l'étudiant:")
for col in features.columns:
    print(f"{col}: {features.iloc[student_index][col]}")

print("\nDécision du modèle:", "Admis" if model.predict(features.iloc[student_index:student_index+1])[0] == 1 else "Non admis")
print("Probabilité d'admission:", round(model.predict_proba(features.iloc[student_index:student_index+1])[0][1], 3))


# In[29]:


# Définition des plages de valeurs acceptables pour chaque feature
feature_ranges = {
    'gre_score': [260, 340],  # Scores GRE réalistes
    'gpa': [2.0, 4.0],        # GPA réalistes
    'research_exp': ['yes', 'no'],
    'recommendation_strength': ['strong', 'medium', 'weak']
}

# Configuration des features à varier avec leurs contraintes métier
permitted_range = {
    'gre_score': [260, 340],
    'gpa': [2.0, 4.0],
    'research_exp': ['yes', 'no'],
    'recommendation_strength': ['strong', 'medium', 'weak']
}

# Définition des changements autorisés
features_to_vary = [
    'gre_score',
    'gpa',
    'research_exp',
    'recommendation_strength'
]

# Configuration des contraintes de proximité
proximity_weight = {
    'gre_score': 0.3,      # Changement modéré autorisé
    'gpa': 0.3,            # Changement modéré autorisé
    'research_exp': 1.0,    # Changement binaire
    'recommendation_strength': 1.0  # Changement catégoriel
}

# Génération des explications avec contraintes
dice_exp = exp.generate_counterfactuals(
    features.iloc[student_index:student_index+1],
    total_CFs=3,
    desired_class="opposite",
    features_to_vary=features_to_vary,
    permitted_range=permitted_range,
    proximity_weight=proximity_weight,
    verbose=False
)

# Ajoutons une fonction pour évaluer la pertinence des changements
def evaluate_counterfactual_changes(original, counterfactual):
    changes = {}
    for feature in features_to_vary:
        if original[feature].values[0] != counterfactual[feature].values[0]:
            changes[feature] = {
                'from': original[feature].values[0],
                'to': counterfactual[feature].values[0],
                'realistic': True  # Par défaut
            }
            
            # Vérification de la pertinence des changements
            if feature == 'gre_score':
                diff = abs(counterfactual[feature].values[0] - original[feature].values[0])
                changes[feature]['realistic'] = diff <= 30  # Maximum 30 points d'écart
            elif feature == 'gpa':
                diff = abs(counterfactual[feature].values[0] - original[feature].values[0])
                changes[feature]['realistic'] = diff <= 0.5  # Maximum 0.5 points d'écart
    
    return changes

# Affichage des résultats avec analyse
print("Profil original:")
for col in features_to_vary:
    print(f"{col}: {features.iloc[student_index][col]}")

print("\nExplications contrefactuelles avec analyse de pertinence:")
counterfactuals = dice_exp.cf_examples_list[0].final_cfs_df

for i in range(len(counterfactuals)):
    print(f"\nScénario {i+1}:")
    changes = evaluate_counterfactual_changes(features.iloc[student_index:student_index+1], 
                                            counterfactuals.iloc[i:i+1])
    for feature, change in changes.items():
        print(f"{feature}: {change['from']} → {change['to']}")
        print(f"Changement réaliste : {'Oui' if change['realistic'] else 'Non'}")

# Affichage des résultats bruts
print("\nRésultats bruts:")
dice_exp.visualize_as_dataframe(show_only_changes=True)


# ## Layerwise Relevance Propagation pour l'analyse d'image

# In[34]:


# Imports nécessaires
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# In[35]:


# Chargement des données
iris = load_iris()
X = iris.data
y = iris.target

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Visualisation des distributions
plt.figure(figsize=(12, 6))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.hist(X[:, i], bins=30)
    plt.title(iris.feature_names[i])
plt.tight_layout()
plt.show()


# In[36]:


def init_weights():
    # Initialisation des poids
    w_j_1 = {'w_i1_j1': -0.2, 'w_i2_j1': 0.5, 'w_i3_j1': -0.1, 'w_i4_j1': 0.3}
    w_j_2 = {'w_i1_j2': -0.25, 'w_i2_j2': 0.1, 'w_i3_j2': 0.4, 'w_i4_j2': -0.2}
    w_j_3 = {'w_i1_j3': 0.2, 'w_i2_j3': -0.3, 'w_i3_j3': -0.2, 'w_i4_j3': 0.1}
    w_j_4 = {'w_i1_j4': 0.5, 'w_i2_j4': -0.8, 'w_i3_j4': 0.6, 'w_i4_j4': -0.3}
    w_j_5 = {'w_i1_j5': 0.1, 'w_i2_j5': 0.4, 'w_i3_j5': 0.7, 'w_i4_j5': 0.2}
    
    w_k = {'w_j1_k1': 0.8, 'w_j2_k1': 0.1, 'w_j3_k1': -0.2, 
           'w_j4_k1': -0.4, 'w_j5_k1': 0.6}
    
    return [w_j_1, w_j_2, w_j_3, w_j_4, w_j_5], w_k

# Visualisation des poids
w_j_list, w_k = init_weights()
weights_matrix = np.zeros((4, 5))
for i in range(4):
    for j in range(5):
        weights_matrix[i, j] = w_j_list[j][f'w_i{i+1}_j{j+1}']

plt.figure(figsize=(10, 6))
sns.heatmap(weights_matrix, annot=True, cmap='coolwarm')
plt.title('Poids entre couche d\'entrée et couche cachée')
plt.xlabel('Neurones couche cachée')
plt.ylabel('Caractéristiques d\'entrée')
plt.show()


# In[37]:


def forward_pass(x_input, w_j_list, w_k):
    # Calcul des sorties de la couche cachée
    val_j = []
    for w_j in w_j_list:
        sum_j = (x_input[0] * w_j[f'w_i1_j{len(val_j)+1}'] +
                x_input[1] * w_j[f'w_i2_j{len(val_j)+1}'] +
                x_input[2] * w_j[f'w_i3_j{len(val_j)+1}'] +
                x_input[3] * w_j[f'w_i4_j{len(val_j)+1}'])
        val_j.append(max(0, sum_j))  # ReLU activation
    
    # Calcul de la sortie finale
    val_k = sum(val_j)
    
    return val_j, val_k

# Test sur un exemple
x_example = X_test[0]
val_j, val_k = forward_pass(x_example, w_j_list, w_k)
print("Activations couche cachée:", val_j)
print("Sortie finale:", val_k)


# In[38]:


def compute_relevances(x_input, val_j, val_k, w_j_list, w_k):
    # Relevance de la couche de sortie
    R_k = val_k
    
    # Relevance de la couche cachée
    R_j = val_j
    
    # Relevance de la couche d'entrée
    R_i = np.zeros(4)
    
    for j, w_j in enumerate(w_j_list):
        sum_j_power = sum(w_j[f'w_i{i+1}_j{j+1}']**2 for i in range(4))
        for i in range(4):
            if sum_j_power > 0:
                R_i[i] += (w_j[f'w_i{i+1}_j{j+1}']**2 / sum_j_power) * R_j[j]
    
    return R_i, R_j, R_k

# Calcul et visualisation des relevances
R_i, R_j, R_k = compute_relevances(x_example, val_j, val_k, w_j_list, w_k)

plt.figure(figsize=(10, 5))
plt.bar(iris.feature_names, R_i)
plt.title('Relevances des caractéristiques d\'entrée')
plt.xticks(rotation=45)
plt.show()


# In[39]:


# Vérification de la positivité
positivity = all(r >= 0 for r in R_i) and all(r >= 0 for r in R_j) and R_k >= 0

# Vérification de la conservativité
conservativity_ij = abs(sum(R_i) - sum(R_j)) < 1e-10
conservativity_jk = abs(sum(R_j) - R_k) < 1e-10

print("Propriétés LRP:")
print(f"Positivité: {positivity}")
print(f"Conservativité i-j: {conservativity_ij}")
print(f"Conservativité j-k: {conservativity_jk}")

# Affichage des sommes pour vérification
print("\nSommes des relevances:")
print(f"Somme R_i: {sum(R_i):.6f}")
print(f"Somme R_j: {sum(R_j):.6f}")
print(f"R_k: {R_k:.6f}")


# In[40]:


# Création d'un DataFrame pour l'analyse
import pandas as pd

results_df = pd.DataFrame({
    'Feature': iris.feature_names,
    'Input_Value': x_example,
    'Relevance': R_i,
    'Relative_Contribution': R_i / sum(R_i) * 100
})

print("Analyse des contributions:")
print(results_df.to_string(index=False))

# Visualisation des contributions relatives
plt.figure(figsize=(10, 5))
plt.pie(results_df['Relative_Contribution'], 
        labels=results_df['Feature'],
        autopct='%1.1f%%')
plt.title('Contribution relative de chaque caractéristique')
plt.show()

