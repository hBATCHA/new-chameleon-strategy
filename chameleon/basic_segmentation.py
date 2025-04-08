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

# 📌 Utiliser le bon modèle correspondant aux poids
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
    print("Modèle chargé avec succès")
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



# Affichage amélioré
plt.figure(figsize=(15, 7.5))

# Image originale
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image originale")
plt.axis('off')

# Segmentation colorée
plt.subplot(1, 2, 2)
plt.imshow(colored_mask)
plt.title("Segmentation (avec couleurs Cityscapes)")
plt.axis('off')

# Ajouter une légende pour les classes présentes
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=np.array(cityscapes_colors[class_id])/255)
                  for class_id in classes_present]
plt.legend(legend_elements,
          [cityscapes_classes[class_id] for class_id in classes_present],
          loc='center left',
          bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
