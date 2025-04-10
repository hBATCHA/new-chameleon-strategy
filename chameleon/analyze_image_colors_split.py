"""
================================================================================
 📄 Script : analyze_image_colors_split.py
 📦 Projet : New Chameleon Strategy
 🧠 Objectif :
     Ce script réalise la segmentation sémantique d'une image avec DeepLabV3+
     (MobileNet, pré-entraîné sur Cityscapes) et analyse les couleurs dominantes
     dans certaines classes, en les séparant entre **moitié gauche et moitié droite**.

 🧰 Fonctionnalités :
     - Chargement du modèle via fichiers locaux (checkpoints/)
     - Segmentation sémantique de l’image (classes Cityscapes)
     - Séparation gauche/droite des classes d’intérêt (végétation, ciel, terrain)
     - Extraction des couleurs dominantes (KMeans) par moitié
     - Visualisation complète (image, masque, légende, graphes de couleur)
     - Affichage console détaillé des proportions et des RGB

 📎 Dépendances :
     - torch
     - torchvision
     - numpy
     - opencv-python (cv2)
     - matplotlib
     - scikit-learn

 🗂️ Structure attendue :
     new-chameleon-strategy/
     ├── chameleon/
     │   └── analyze_image_colors_split.py
     ├── checkpoints/
     │   └── best_deeplabv3plus_mobilenet_cityscapes_os16.pth
     ├── samples/
     │   └── image17.jpg
     └── DeepLabV3Plus-Pytorch/
         └── network/, modeling/, etc.

 🧪 Auteur :
     - Hashif Batcha
     - Basé sur DeepLabV3Plus-Pytorch (VainF)
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
from sklearn.cluster import KMeans


model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19)
# Construire un chemin relatif vers le modèle
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


# Charger et afficher l'image d'origine
image_path = os.path.join(base_dir, "samples", "image17.jpg")
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.title("Image d'origine")
plt.show()

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

# Créer et afficher le masque de segmentation
mask = output_tensor.argmax(0).cpu().numpy()

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

# Après la définition de cityscapes_colors, ajoutez un dictionnaire de noms
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

# Pour trouver les classes présentes dans l'image
classes_present = np.unique(mask)
print("\nClasses présentes dans l'image :")
for class_id in classes_present:
    color = cityscapes_colors[class_id]
    class_name = cityscapes_classes[class_id]
    print(f"Classe {class_id} ({class_name}): RGB{color}")

# Pour calculer le pourcentage de chaque classe
total_pixels = mask.size
print("\nPourcentage de chaque classe :")
for class_id in classes_present:
    n_pixels = np.sum(mask == class_id)
    percentage = (n_pixels / total_pixels) * 100
    print(f"{cityscapes_classes[class_id]}: {percentage:.2f}%")

# Créer une image segmentée colorée
colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for class_id, color in enumerate(cityscapes_colors):
    colored_mask[mask == class_id] = color


def extract_dominant_colors_split(image, mask, class_id, n_colors=2):
    # Redimensionner l'image aux dimensions du masque
    h_mask, w_mask = mask.shape
    image_resized = cv2.resize(image, (w_mask, h_mask))

    # Créer un masque booléen pour la classe
    class_mask = mask == class_id

    # Si aucun pixel n'appartient à cette classe, retourner None
    if not np.any(class_mask):
        return None, None, None, None

    # Calculer le point médian pour la séparation
    mid_w = w_mask // 2

    # Créer les masques pour la partie gauche et droite
    left_mask = class_mask.copy()
    left_mask[:, mid_w:] = False
    right_mask = class_mask.copy()
    right_mask[:, :mid_w] = False

    # Extraire les pixels pour chaque partie
    left_pixels = image_resized[left_mask]
    right_pixels = image_resized[right_mask]

    left_colors = left_proportions = right_colors = right_proportions = None

    # Traiter la partie gauche si elle contient des pixels
    if len(left_pixels) > 0:
        kmeans_left = KMeans(n_clusters=n_colors, random_state=42)
        kmeans_left.fit(left_pixels)
        left_colors = kmeans_left.cluster_centers_.astype(int)
        left_proportions = np.bincount(kmeans_left.labels_) / len(kmeans_left.labels_)

    # Traiter la partie droite si elle contient des pixels
    if len(right_pixels) > 0:
        kmeans_right = KMeans(n_clusters=n_colors, random_state=42)
        kmeans_right.fit(right_pixels)
        right_colors = kmeans_right.cluster_centers_.astype(int)
        right_proportions = np.bincount(kmeans_right.labels_) / len(kmeans_right.labels_)

    return left_colors, left_proportions, right_colors, right_proportions

def visualize_colors_split(image, mask, classes_of_interest, classes_present):
    # Calculer le nombre de lignes nécessaires
    n_rows = len(classes_of_interest) + 1  # +1 pour l'image originale et la segmentation
    n_cols = 3

    plt.figure(figsize=(20, n_rows * 4))

    # Image originale
    plt.subplot(n_rows, n_cols, 1)
    plt.imshow(image)
    plt.title("Image originale")
    plt.axis('off')

    # Segmentation colorée
    plt.subplot(n_rows, n_cols, 2)
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(cityscapes_colors):
        colored_mask[mask == class_id] = color
    plt.imshow(colored_mask)
    plt.title("Segmentation")
    plt.axis('off')

    # Ajout de la légende de toutes les classes présentes dans le masque
    plt.subplot(n_rows, n_cols, 3)
    plt.axis('off')
    y_pos = 0.95  # Position de départ en haut
    plt.text(0.1, y_pos + 0.05, "Classes présentes:", fontsize=12, fontweight='bold')

    # Trouver toutes les classes uniques dans le masque
    unique_classes = np.unique(mask)

    # Créer des patches de couleur pour chaque classe dans le masque
    for class_id in unique_classes:
        y_pos -= 0.1  # Espace entre chaque classe
        color = np.array(cityscapes_colors[class_id], dtype=np.float32) / 255.0
        plt.plot([0.1, 0.2], [y_pos, y_pos], color=color, linewidth=10)

        # Ajouter une indication si la classe fait partie des classes d'intérêt
        class_name = f"Classe {class_id}"
        if class_id in classes_of_interest:
            class_name = f"{classes_of_interest[class_id]} (Classe {class_id})*"

        plt.text(0.25, y_pos, class_name, fontsize=10, verticalalignment='center')

    # Ajouter une note pour les classes d'intérêt
    y_pos -= 0.15
    plt.text(0.1, y_pos, "* Classes analysées pour les couleurs", fontsize=8, style='italic')
    plt.title("Légende")

    # Commencer à partir de la deuxième ligne
    current_plot = n_cols + 1

    for class_id, class_name in classes_of_interest.items():
        if class_id in classes_present:
            left_colors, left_proportions, right_colors, right_proportions = extract_dominant_colors_split(image, mask,
                                                                                                           class_id)

            # Subplot pour cette classe
            plt.subplot(n_rows, n_cols, current_plot)
            plt.title(f"{class_name} - Couleurs dominantes")

            y_pos = 0
            bar_height = 0.3

            # Afficher les couleurs de la partie gauche
            if left_colors is not None:
                plt.text(-0.1, y_pos + 1.5, "Partie gauche:", fontsize=10)
                for i, (color, prop) in enumerate(zip(left_colors, left_proportions)):
                    plt.barh(y_pos, prop, height=bar_height, color=color / 255)
                    plt.text(prop + 0.02, y_pos, f'RGB{i + 1} : {color} - Proportion : {prop * 100:.1f}%',
                             va='center', fontsize=8)
                    y_pos += bar_height + 0.1

            y_pos += 0.5

            # Afficher les couleurs de la partie droite
            if right_colors is not None:
                plt.text(-0.1, y_pos + 1.5, "Partie droite:", fontsize=10)
                for i, (color, prop) in enumerate(zip(right_colors, right_proportions)):
                    plt.barh(y_pos, prop, height=bar_height, color=color / 255)
                    plt.text(prop + 0.02, y_pos, f'RGB{i + 1} : {color} - Proportion : {prop * 100:.1f}%',
                             va='center', fontsize=8)
                    y_pos += bar_height + 0.1

            plt.xlim(0, 1.2)
            plt.axis('off')
            current_plot += 3

    plt.tight_layout()
    plt.show()


# Classes d'intérêt
classes_of_interest = {
    8: 'végétation',
    9: 'terrain',
    10: 'ciel'
}

# Visualiser les résultats avec la nouvelle fonction
visualize_colors_split(image, mask, classes_of_interest, classes_present)

# Afficher les statistiques détaillées dans la console
print("\nAnalyse détaillée des couleurs dominantes par moitié :")
for class_id, class_name in classes_of_interest.items():
    if class_id in classes_present:
        print(f"\n{class_name.upper()} :")
        left_colors, left_proportions, right_colors, right_proportions = extract_dominant_colors_split(image, mask,
                                                                                                       class_id)

        print("PARTIE GAUCHE :")
        if left_colors is not None:
            for i, (color, prop) in enumerate(zip(left_colors, left_proportions)):
                print(f"- Couleur RGB {i + 1} : {color} - Proportion : {prop * 100:.1f}%")

        print("\nPARTIE DROITE :")
        if right_colors is not None:
            for i, (color, prop) in enumerate(zip(right_colors, right_proportions)):
                print(f"- Couleur RGB {i + 1} : {color} - Proportion : {prop * 100:.1f}%")
