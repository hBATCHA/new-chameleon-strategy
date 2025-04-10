"""
================================================================================
 📄 Module : color_analysis_utils.py
 📦 Projet : New Chameleon Strategy – Phase finale image
 🧠 Objectif :
     Ce module contient les fonctions de traitement utilisées pour extraire
     les couleurs dominantes par classe après segmentation d’une image.

 🧰 Fonctionnalités :
     - Prétraitement de l’image (resize, normalisation)
     - Application de KMeans par classe
     - Interpolation optionnelle des couleurs
     - Option de tri basé sur la position spatiale des pixels

 📎 Utilisé par :
     - analyze_image.py

 📎 Dépendances :
     - torch, torchvision
     - numpy, opencv-python
     - scikit-learn

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

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import network

# Importer les fonctions de visualisation améliorée
from visualization_utils import (
    visualize_segmentation_results,
    extract_dominant_colors,
    extract_dominant_colors_split,
    create_results_table,
    create_dominant_color_images,
    create_dominant_color_images_with_interpolation,
    create_complete_visualization, create_dominant_color_images_with_position,
    create_dominant_color_images_with_interpolation_and_position,
)

# Charger le modèle DeepLabV3+
model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19)
model_path = os.path.join(base_dir, "checkpoints", "best_deeplabv3plus_mobilenet_cityscapes_os16.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'], strict=True)
    else:
        print("Erreur : Le checkpoint ne contient pas 'model_state'")
    print("Modèle chargé avec succès !")
except Exception as e:
    print(f"Erreur lors du chargement des poids : {e}")

model.to(device)
model.eval()


def preprocess_image(image):
    # Calculer le ratio pour conserver les proportions
    h, w = image.shape[:2]
    ratio = 512 / h
    new_w = int(w * ratio)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, new_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# Définir une palette de couleurs pour les classes Cityscapes
cityscapes_colors = [
    [128, 64, 128],  # route
    [244, 35, 232],  # trottoir
    [70, 70, 70],  # bâtiment
    [102, 102, 156],  # mur
    [190, 153, 153],  # clôture
    [153, 153, 153],  # poteau
    [250, 170, 30],  # feu de circulation
    [220, 220, 0],  # panneau de signalisation
    [107, 142, 35],  # végétation
    [152, 251, 152],  # terrain
    [70, 130, 180],  # ciel
    [220, 20, 60],  # personne
    [255, 0, 0],  # rider
    [0, 0, 142],  # voiture
    [0, 0, 70],  # camion
    [0, 60, 100],  # bus
    [0, 80, 100],  # train
    [0, 0, 230],  # moto
    [119, 11, 32]  # vélo
]

# Dictionnaire des noms de classes
cityscapes_classes = {
    0: 'route',
    1: 'trottoir',
    2: 'bâtiment',
    3: 'mur',
    4: 'clôture',
    5: 'poteau',
    6: 'feu de circulation',
    7: 'panneau de signalisation',
    8: 'végétation',
    9: 'terrain',
    10: 'ciel',
    11: 'personne',
    12: 'rider',
    13: 'voiture',
    14: 'camion',
    15: 'bus',
    16: 'train',
    17: 'moto',
    18: 'vélo'
}


# Modification de la fonction analyze_image pour utiliser les nouvelles fonctions
def analyze_image(image_path, output_dir="./results", interpolation=True, interpolation_strength=3,
                  use_position_based=True):
    """
    Analyse une image avec DeepLabV3+ et génère une visualisation détaillée
    ainsi que des images de synthèse des couleurs dominantes dans un dossier spécifique à cette image

    Args:
        image_path: Chemin vers l'image à analyser
        output_dir: Répertoire principal pour sauvegarder les résultats
        interpolation: Activer l'interpolation entre les bandes de couleurs
        interpolation_strength: Force de l'interpolation (plus élevé = transitions plus douces)
        use_position_based: Utiliser l'extraction de couleurs basée sur la position

    Returns:
        fig_complete: Figure complète avec toutes les visualisations
    """
    import os

    # Créer le répertoire principal de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Extraire le nom de base de l'image (sans extension)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Créer un sous-dossier spécifique pour cette image
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Préparer l'image et faire la prédiction
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(input_tensor)

        if isinstance(output, dict) and 'out' in output:
            output_tensor = output['out'][0]
        elif isinstance(output, torch.Tensor):
            output_tensor = output[0]
        else:
            raise ValueError(f"Structure de sortie inattendue : {type(output)}")

    # Créer le masque de segmentation
    mask = output_tensor.argmax(0).cpu().numpy()

    # Définir les classes d'intérêt
    classes_of_interest = {
        8: 'végétation',
        9: 'terrain',
        10: 'ciel'
    }

    # Définir le nombre de couleurs à extraire pour chaque classe
    colors_per_class = {
        8: {'global': 5, 'split': 3},  # végétation: 5 couleurs globales, 3 par côté
        9: {'global': 3, 'split': 3},  # terrain: 3 couleurs globales, 3 par côté
        10: {'global': 2, 'split': 2},  # ciel: 2 couleurs globales, 2 par côté
    }

    # Obtenir les classes présentes dans l'image
    classes_present = np.unique(mask)

    # 1. Générer la visualisation améliorée des couleurs dominantes
    fig_detail = visualize_segmentation_results(
        image, mask, cityscapes_colors, cityscapes_classes,
        classes_of_interest, colors_per_class
    )

    # Sauvegarder la figure détaillée dans le sous-dossier de l'image
    fig_detail.savefig(f"{image_output_dir}/analyse_detaillee.png", dpi=300, bbox_inches='tight')
    plt.close(fig_detail)

    # 2. Créer un rapport textuel
    report = create_results_table(
        image, mask, classes_of_interest, classes_present, cityscapes_classes, colors_per_class
    )

    # Sauvegarder le rapport dans le sous-dossier de l'image
    with open(f"{image_output_dir}/rapport.txt", "w") as f:
        f.write(report)

    # 3. Créer les images de synthèse des couleurs dominantes avec ou sans interpolation
    # En utilisant les nouvelles fonctions basées sur la position si demandé
    if use_position_based:
        if interpolation:
            global_img, split_img = create_dominant_color_images_with_interpolation_and_position(
                image, mask, cityscapes_classes, colors_per_class,
                image_output_dir, "synthese", debug=True,
                interpolation_strength=interpolation_strength
            )
        else:
            global_img, split_img = create_dominant_color_images_with_position(
                image, mask, cityscapes_classes, colors_per_class,
                image_output_dir, "synthese", debug=True
            )
    else:
        # Utiliser les fonctions originales
        if interpolation:
            global_img, split_img = create_dominant_color_images_with_interpolation(
                image, mask, cityscapes_classes, colors_per_class,
                image_output_dir, "synthese", debug=True,
                interpolation_strength=interpolation_strength
            )
        else:
            global_img, split_img = create_dominant_color_images(
                image, mask, cityscapes_classes, colors_per_class,
                image_output_dir, "synthese", debug=True
            )

    # 4. Créer une visualisation complète avec toutes les images
    fig_complete = create_complete_visualization(
        image, mask, global_img, split_img,
        cityscapes_colors, cityscapes_classes,
        classes_of_interest, colors_per_class,
        image_output_dir, "visualisation"
    )

    # 5. Créer aussi une figure séparée pour les images de synthèse uniquement
    fig_colors = plt.figure(figsize=(16, 8))

    # Image de synthèse globale
    ax1 = fig_colors.add_subplot(1, 2, 1)
    ax1.imshow(global_img)
    ax1.set_title("Couleurs dominantes globales", fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Image de synthèse gauche/droite
    ax2 = fig_colors.add_subplot(1, 2, 2)
    ax2.imshow(split_img)
    ax2.set_title("Couleurs dominantes gauche/droite", fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Ajouter information sur l'interpolation et la méthode utilisée dans le titre
    title = "Synthèse des couleurs dominantes"
    if use_position_based:
        title += " (basée sur la position)"
    if interpolation:
        title += f" avec interpolation (force: {interpolation_strength})"

    fig_colors.suptitle(title, fontsize=16, fontweight='bold')

    # Sauvegarder la figure des couleurs dominantes
    fig_colors.savefig(f"{image_output_dir}/synthese_couleurs.png", dpi=300, bbox_inches='tight')
    plt.close(fig_colors)

    print(f"Analyse terminée pour {image_path}")
    print(f"Résultats sauvegardés dans {image_output_dir}")

    # Retourner la figure complète
    return fig_complete


# Mise à jour de la fonction analyze_all_images
def analyze_all_images(input_dir, output_dir="resultats", interpolation=True,
                       interpolation_strength=3, use_position_based=True):
    """
    Analyse toutes les images d'un dossier en créant un sous-dossier pour chaque image

    Args:
        input_dir: Répertoire contenant les images à analyser
        output_dir: Répertoire principal pour sauvegarder les résultats
        interpolation: Activer l'interpolation entre les bandes de couleurs
        interpolation_strength: Force de l'interpolation
        use_position_based: Utiliser l'extraction de couleurs basée sur la position
    """
    import os

    # Créer le répertoire principal de sortie
    os.makedirs(output_dir, exist_ok=True)

    # Extensions d'images supportées
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Récupérer tous les fichiers image du dossier
    image_files = []
    for filename in os.listdir(input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            image_files.append(os.path.join(input_dir, filename))

    # Analyser chaque image
    for img_file in image_files:
        print(f"Analyse de {img_file}...")
        analyze_image(img_file, output_dir, interpolation,
                      interpolation_strength, use_position_based)

    print(f"Analyse terminée pour {len(image_files)} images")
    print(f"Tous les résultats sont organisés dans des sous-dossiers de {output_dir}")


# Script principal modifié
if __name__ == "__main__":
    # Vous pouvez changer le chemin d'accès à l'image ici
    image_path = "images/image17.jpg"

    # Paramètres d'analyse
    use_interpolation = True
    interpolation_force = 3
    use_position_ordering = True  # Activer l'extraction basée sur la position

    fig = analyze_image(
        image_path,
        interpolation=use_interpolation,
        interpolation_strength=interpolation_force,
        use_position_based=use_position_ordering
    )

    plt.show()