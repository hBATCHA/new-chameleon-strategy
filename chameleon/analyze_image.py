"""
================================================================================
 📄 Script : analyze_image.py
 📦 Projet : New Chameleon Strategy – Phase finale image
 🧠 Objectif :
     Ce script exécute une analyse complète d’une image :
     - Segmentation sémantique avec DeepLabV3+ (MobileNet, Cityscapes)
     - Extraction des couleurs dominantes par classe
     - Visualisation avancée et export automatique des résultats

 🧰 Fonctionnalités :
     - Chargement dynamique du modèle (checkpoints/)
     - Lecture d’image et prétraitement
     - KMeans appliqué par classe (avec options : interpolation, position, etc.)
     - Affichage + enregistrement des résultats dans `results/`
     - Structure modulaire basée sur `color_analysis_utils.py` et `visualization_utils.py`

 📎 Dépendances :
     - torch, torchvision
     - numpy, opencv-python, matplotlib
     - scikit-learn

 📎 Modules utilisés :
     - color_analysis_utils.py
     - visualization_utils.py

 🧪 Auteur : Hashif Batcha
================================================================================
"""

import sys
import os

# Récupérer le chemin du projet (racine)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ajouter DeepLabV3Plus-Pytorch au sys.path pour pouvoir importer "network"
deeplab_path = os.path.join(base_dir, "DeepLabV3Plus-Pytorch")
sys.path.append(deeplab_path)

import time
from matplotlib import pyplot as plt

# Importer la fonction d'analyse
from color_analysis_utils import analyze_image

# Paramètres de configuration
image_path = os.path.join(base_dir, "samples", "image17.jpg")
output_dir = os.path.join(base_dir, "results")

# Paramètres d'interpolation
activer_interpolation = True     # Mettre True pour activer l'interpolation, False pour la désactiver
force_interpolation = 10         # Valeur entre 1 et 20 (plus la valeur est élevée, plus la transition est douce)

# Nouveau paramètre pour activer l'ordonnancement basé sur la position
ordonner_par_position = True     # Mettre True pour que les couleurs reflètent leur position dans l'image originale

# Démarrer le chronomètre
start_time = time.time()

# Exécuter l'analyse avec les paramètres
fig = analyze_image(
    image_path,
    output_dir=output_dir,
    interpolation=activer_interpolation,
    interpolation_strength=force_interpolation,
    use_position_based=ordonner_par_position
)

# Arrêter le chronomètre et afficher le temps d'exécution
end_time = time.time()
print(f"Temps d'exécution : {end_time - start_time:.2f} secondes")

# Afficher les résultats
plt.show()  # Pour afficher immédiatement les résultats