"""
================================================================================
 📄 Script : analyze_image_colors.py
 📦 Projet : New Chameleon Strategy
 🧠 Objectif :
     Ce script réalise la segmentation sémantique d'une image avec DeepLabV3+
     (MobileNet backbone, entraîné sur Cityscapes) et extrait les couleurs
     dominantes pour certaines classes d’intérêt (ex : végétation, ciel, terrain).

 🧰 Fonctionnalités :
     - Chargement dynamique du modèle depuis checkpoints/
     - Segmentation de l’image d’entrée (ex : samples/image17.jpg)
     - Génération d’un masque coloré selon la palette Cityscapes
     - Analyse des classes présentes dans l’image
     - Extraction des couleurs dominantes (KMeans) pour les classes cibles
     - Visualisation complète (image originale, segmentation, couleurs dominantes)
     - Affichage des proportions par classe et statistiques console

 📎 Dépendances :
     - torch
     - torchvision
     - numpy
     - opencv-python (cv2)
     - matplotlib
     - scikit-learn

 🗂️ Organisation du projet recommandée :
     new-chameleon-strategy/
     ├── chameleon/
     │   └── analyze_image_colors.py
     ├── checkpoints/
     │   └── best_deeplabv3plus_mobilenet_cityscapes_os16.pth
     ├── samples/
     │   └── image17.jpg
     └── DeepLabV3Plus-Pytorch/
         └── network/, modeling/, etc.

 🧪 Auteurs :
     - Hashif Batcha
     - Basé sur DeepLabV3Plus-Pytorch (par VainF)
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


def extract_dominant_colors(image, mask, class_id, n_colors=7):
    # Redimensionner l'image aux dimensions du masque
    h_mask, w_mask = mask.shape
    image_resized = cv2.resize(image, (w_mask, h_mask))

    # Créer un masque booléen pour la classe
    class_mask = mask == class_id

    # Si aucun pixel n'appartient à cette classe, retourner None
    if not np.any(class_mask):
        return None, None

    # Extraire les pixels de l'image correspondant à la classe
    pixels = image_resized[class_mask]

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)

    # Obtenir les couleurs et leur proportion
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Calculer les proportions
    proportions = np.bincount(labels) / len(labels)

    return colors, proportions

# Classes d'intérêt
classes_of_interest = {
    8: 'végétation',
    9: 'terrain',
    10: 'ciel'
}

# Créer une figure plus grande pour accommoder tous les graphiques
plt.figure(figsize=(20, 10))

# Image originale
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Image originale")
plt.axis('off')

# Segmentation colorée
plt.subplot(1, 3, 2)
plt.imshow(colored_mask)
plt.title("Segmentation (avec couleurs Cityscapes)")
plt.axis('off')

# Ajouter une légende pour les classes présentes
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=np.array(cityscapes_colors[class_id]) / 255)
                   for class_id in classes_present]
plt.legend(legend_elements,
           [cityscapes_classes[class_id] for class_id in classes_present],
           loc='center left',
           bbox_to_anchor=(1, 0.5))

# Nouveau subplot pour les couleurs dominantes
plt.subplot(1, 3, 3)
plt.axis('off')
plt.title("Couleurs dominantes par classe", loc='center', pad=20)

# Réduire l'espacement initial et entre les éléments
y_position = 0.95
class_spacing = 0.05  # Espacement entre les classes
color_spacing = 0.04  # Espacement entre les couleurs

for class_id, class_name in classes_of_interest.items():
    if class_id in classes_present:
        colors, proportions = extract_dominant_colors(image, mask, class_id)
        if colors is not None:
            # Afficher le nom de la classe
            plt.text(0.1, y_position, f"{class_name}:", fontsize=10, transform=plt.gca().transAxes)
            y_position -= class_spacing

            # Afficher les couleurs et leurs proportions
            for i, (color, prop) in enumerate(zip(colors, proportions)):
                # Créer un rectangle de couleur
                rect = plt.Rectangle((0.1, y_position - 0.03), 0.1, 0.03,
                                     facecolor=color / 255)
                plt.gca().add_patch(rect)

                # Afficher le pourcentage avec une taille de police plus petite
                plt.text(0.25, y_position - 0.03,
                         f"Couleur RGB {i+1} : {color} - Proportion : ({prop * 100:.1f}%)",
                         fontsize=7, transform=plt.gca().transAxes)
                y_position -= color_spacing

            # Ajouter un petit espace supplémentaire entre les classes
            y_position -= 0.06

plt.tight_layout()
plt.show()

# Afficher les statistiques détaillées dans la console
print("\nAnalyse détaillée des couleurs dominantes :")
for class_id, class_name in classes_of_interest.items():
    if class_id in classes_present:
        print(f"\n{class_name.upper()} :")
        colors, proportions = extract_dominant_colors(image, mask, class_id)
        if colors is not None:
            for i, (color, prop) in enumerate(zip(colors, proportions)):
                print(f"- Couleur RGB {i+1} : {color} - Proportion : {prop * 100:.1f}%")
