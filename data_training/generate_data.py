import pandas as pd
import numpy as np
import random

# 1. Configuration
n_students = 6000  # Nombre de lignes demandé
filieres_bac = ['SM-A', 'SM-B', 'PC', 'SVT', 'Eco-Gestion']

# Initialisation des graines aléatoires pour avoir les mêmes résultats à chaque lancement
np.random.seed(42)
random.seed(42)

# 2. Génération des notes de base
# On utilise une distribution normale (Gaussienne) pour simuler la réalité
data = {
    'Moyenne_Generale': np.random.normal(13, 3, n_students).clip(10, 20),
    'Note_Maths': [],
    'Note_Physique': [],
    'Note_Francais': [],
    'Filiere_Bac': [random.choice(filieres_bac) for _ in range(n_students)]
}

# 3. Ajustement des notes selon la filière (Logique Métier)
for i in range(n_students):
    filiere = data['Filiere_Bac'][i]
    
    # Logique : Un SM aura généralement de meilleures notes en sciences qu'un SVT ou Eco
    if 'SM' in filiere:
        data['Note_Maths'].append(np.random.normal(16, 2))
        data['Note_Physique'].append(np.random.normal(15, 2))
    elif 'PC' in filiere:
        data['Note_Maths'].append(np.random.normal(14, 2))
        data['Note_Physique'].append(np.random.normal(16, 2))
    else:
        # Pour SVT et Eco, les notes en maths/physique sont souvent plus basses ou plus dispersées
        data['Note_Maths'].append(np.random.normal(12, 3))
        data['Note_Physique'].append(np.random.normal(11, 3))
    
    # Le français est généré indépendamment de la filière scientifique
    data['Note_Francais'].append(np.random.normal(13, 3))

# 4. Création du DataFrame Pandas
df = pd.DataFrame(data)

# NETTOYAGE CRUCIAL : On force toutes les notes entre 0 et 20
cols_notes = ['Note_Maths', 'Note_Physique', 'Note_Francais', 'Moyenne_Generale']
df[cols_notes] = df[cols_notes].clip(0, 20)

# Arrondir à 2 décimales pour faire "vrai bulletin"
df[cols_notes] = df[cols_notes].round(2)

# 5. Fonction d'étiquetage (Target)
def recommander_filiere(row):
    # Score scientifique simple
    score_science = (row['Note_Maths'] + row['Note_Physique']) / 2
    
    # Règles de décision (Hiérarchie des choix au Maroc)
    if row['Moyenne_Generale'] >= 16 and score_science > 16:
        return 'Medecine / Pharmacie'
    elif row['Note_Maths'] > 15 and row['Moyenne_Generale'] > 14:
        return 'CPGE (Prépas)'
    elif row['Note_Maths'] > 13 and row['Note_Physique'] > 13:
        return 'ENSA / ENSAM'
    elif row['Filiere_Bac'] == 'Eco-Gestion' and row['Moyenne_Generale'] > 12:
        return 'ENCG'
    elif row['Moyenne_Generale'] > 11:
        return 'FST'
    else:
        return 'Faculté (Licence Fondamentale)'

# Application de la fonction
df['Recommendation'] = df.apply(recommander_filiere, axis=1)

# 6. Sauvegarde et Vérification
file_name = 'dataset_orientation_maroc_6000.csv'
df.to_csv(file_name, index=False)

print(f"✅ Fichier '{file_name}' généré avec succès !")
print(f"Dimension : {df.shape}")
print("\nDistribution des classes (Target) :")
print(df['Recommendation'].value_counts())