"""
================================================================================
 📄 Script : analyze_video_frames.py
 📦 Projet : New Chameleon Strategy – Partie Vidéo (V1)
 🧠 Objectif :
     Ce script applique une segmentation sémantique DeepLabV3+ frame par frame
     sur une vidéo locale, puis génère une nouvelle vidéo contenant les masques
     segmentés colorés selon la palette Cityscapes.

 🧰 Fonctionnalités :
     - Chargement du modèle DeepLabV3+ (MobileNet) depuis checkpoints/
     - Lecture d'une vidéo depuis le dossier `video_inputs/`
     - Traitement de chaque frame avec segmentation sémantique
     - Génération d'une vidéo en sortie avec les masques colorés
     - Création automatique d'un dossier `video_outputs/`
     - Suivi de la progression et du temps d'exécution

 📎 Répertoires utilisés :
     - `video_inputs/` : contient les vidéos d'entrée à traiter
     - `video_outputs/` : vidéos générées automatiquement (non versionnées)

 📎 Dépendances :
     - torch, torchvision
     - opencv-python
     - numpy
     - torchvision.transforms
     - network (module local de DeepLabV3+)

 🧪 Auteur : Hashif Batcha
================================================================================
"""

import torch
import cv2
import numpy as np
import time
import os
import sys

# Récupérer le chemin du projet (racine)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Ajouter DeepLabV3Plus-Pytorch au sys.path pour pouvoir importer "network"
deeplab_path = os.path.join(base_dir, "DeepLabV3Plus-Pytorch")
sys.path.append(deeplab_path)

import network

class VideoSplitSynthesizer:
    def __init__(self):
        """Initialisation du synthétiseur d'images split"""

        # Pour Apple Silicon M4 Max
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Utilisation du GPU Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Utilisation de CUDA")
        else:
            self.device = torch.device("cpu")
            print("Utilisation du CPU")

        # Chargement du modèle DeepLabV3+
        self.model = network.modeling.__dict__["deeplabv3plus_mobilenet"](num_classes=19)
        model_path = os.path.join(base_dir, "checkpoints", "best_deeplabv3plus_mobilenet_cityscapes_os16.pth")

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state'], strict=True)
                print("Modèle chargé avec succès!")
            else:
                print("Erreur: Le checkpoint ne contient pas 'model_state'")
        except Exception as e:
            print(f"Erreur lors du chargement des poids: {e}")

        self.model.to(self.device)
        self.model.eval()

        # Définir les classes d'intérêt
        self.classes_of_interest = {
            8: 'végétation',
            9: 'terrain',
            10: 'ciel'
        }

        # Ordre vertical pour l'affichage
        self.vertical_order = [10, 8, 9]  # [ciel, végétation, terrain]

        # Configuration des couleurs par classe
        self.colors_per_class = {
            8: {'split': 3},  # végétation: 3 couleurs par côté
            9: {'split': 3},  # terrain: 3 couleurs par côté
            10: {'split': 2},  # ciel: 2 couleurs par côté
        }

        # Paramètres d'optimisation
        self.use_interpolation = True
        self.interpolation_strength = 10  # Augmenté pour avoir des transitions plus douces (comme dans la version statique)
        self.use_position_based = True  # Utiliser l'extraction basée sur la position

        # Nouveau: préréglages de qualité
        self.quality_preset = "balanced"  # "performance", "balanced", "quality"
        self.update_quality_settings()

        # Paramètres de debug
        self.save_debug_images = False
        self.debug_dir = os.path.join(base_dir, "debug_images")

        # Créer le dossier de debug si nécessaire
        if self.save_debug_images:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Paramètres du lissage temporel
        self.temporal_smoothing = True  # Activer par défaut
        self.smoothing_frames = 5  # Nombre de frames à considérer pour le lissage
        self.smoothing_weight = 0.6  # Poids de la frame courante (0-1)
        self.previous_frames = []  # Stockage des frames précédentes
        self.previous_left_colors = {}  # Stockage des couleurs gauche précédentes par classe
        self.previous_right_colors = {}  # Stockage des couleurs droite précédentes par classe


    def update_quality_settings(self):
        """Met à jour les paramètres en fonction du préréglage de qualité choisi"""
        if self.quality_preset == "performance":
            self.processing_resolution = (384, 384)
            self.sample_size_factor = 0.25  # 25% des pixels
            self.kmeans_inits = 1
            self.kmeans_max_iter = 100
            self.min_band_height = 3  # Hauteur minimale d'une bande en pixels
        elif self.quality_preset == "balanced":
            self.processing_resolution = (448, 448)
            self.sample_size_factor = 0.4  # 40% des pixels
            self.kmeans_inits = 2
            self.kmeans_max_iter = 150
            self.min_band_height = 4  # Hauteur minimale d'une bande en pixels
        else:  # "quality"
            self.processing_resolution = (512, 512)
            self.sample_size_factor = 0.6  # 60% des pixels
            self.kmeans_inits = 3
            self.kmeans_max_iter = 200
            self.min_band_height = 5  # Hauteur minimale d'une bande en pixels

        print(f"Qualité: {self.quality_preset}, Résolution: {self.processing_resolution}")

    def preprocess_image(self, image):
        """Prétraite l'image pour le modèle"""
        # Redimensionner l'image à la résolution de traitement
        resized = cv2.resize(image, self.processing_resolution)

        # Convertir en tensor PyTorch
        input_tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()

        # Normaliser
        input_tensor = input_tensor.div(255.0)
        input_tensor = input_tensor.sub(torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        input_tensor = input_tensor.div(torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))

        return input_tensor.unsqueeze(0)

    def extract_dominant_colors_with_position(self, image, boolean_mask, n_colors=3):
        """Extraire les couleurs dominantes en gardant l'information de position verticale"""
        # Si aucun pixel n'est sélectionné, retourner None
        if not np.any(boolean_mask):
            return None, None

        # Extraire les pixels et leurs positions
        h, w = boolean_mask.shape
        pixels = []
        positions_y = []

        for y in range(h):
            for x in range(w):
                if boolean_mask[y, x]:
                    pixels.append(image[y, x])
                    positions_y.append(y)

        pixels = np.array(pixels)
        positions_y = np.array(positions_y)

        # Sous-échantillonnage pour accélérer le clustering si beaucoup de pixels
        if len(pixels) > 10000:
            sample_size = min(int(10000 * self.sample_size_factor), len(pixels) // 2)
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            pixels = pixels[indices]
            positions_y = positions_y[indices]

        # Éviter une erreur si le nombre de pixels est inférieur à n_colors
        if len(pixels) < n_colors:
            n_colors = max(1, len(pixels))

        # Appliquer KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42,
                        n_init=self.kmeans_inits, max_iter=self.kmeans_max_iter)
        kmeans.fit(pixels)

        # Obtenir les couleurs et leur proportion
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Calculer les proportions
        proportions = np.bincount(labels) / len(labels)

        # Calculer la position verticale moyenne pour chaque cluster
        cluster_positions = []
        for i in range(n_colors):
            cluster_mask = labels == i
            if np.any(cluster_mask):
                avg_pos = np.median(positions_y[cluster_mask])
                cluster_positions.append(avg_pos)
            else:
                cluster_positions.append(0)

        # Trier par position verticale (de haut en bas)
        idx = np.argsort(cluster_positions)
        colors = colors[idx]
        proportions = proportions[idx]

        return colors, proportions

    def interpolate_colors(self, color1, color2, num_steps):
        """Interpolation linéaire entre deux couleurs"""
        r = np.linspace(color1[0], color2[0], num_steps)
        g = np.linspace(color1[1], color2[1], num_steps)
        b = np.linspace(color1[2], color2[2], num_steps)
        return np.column_stack((r, g, b))

    def create_split_image(self, image, mask, debug_info=None):
        """Crée uniquement l'image split à partir de l'image et du masque"""
        h_img, w_img = image.shape[:2]
        h_mask, w_mask = mask.shape

        # Créer une version colorée du masque pour le debug
        if self.save_debug_images and debug_info is not None:
            # Définir des couleurs pour les classes d'intérêt
            mask_colors = np.zeros((h_mask, w_mask, 3), dtype=np.uint8)
            for class_id in self.classes_of_interest:
                if class_id == 8:  # végétation
                    mask_colors[mask == class_id] = [0, 255, 0]  # vert
                elif class_id == 9:  # terrain
                    mask_colors[mask == class_id] = [165, 42, 42]  # marron
                elif class_id == 10:  # ciel
                    mask_colors[mask == class_id] = [135, 206, 235]  # bleu ciel

            cv2.imwrite(f"{self.debug_dir}/mask_{debug_info['frame_id']:04d}.jpg", mask_colors)

        # Calculer les points milieux
        mid_w_img = w_img // 2
        mid_w_mask = w_mask // 2

        # Créer des masques pour les côtés gauche et droit
        mask_left = np.zeros_like(mask, dtype=bool)
        mask_right = np.zeros_like(mask, dtype=bool)
        mask_left[:, :mid_w_mask] = True
        mask_right[:, mid_w_mask:] = True

        # Initialiser les variables pour stocker les données
        pixels_left = {}
        pixels_right = {}
        classes_present_left = []
        classes_present_right = []
        left_colors = {}
        right_colors = {}
        left_props = {}
        right_props = {}

        # Extraire les classes et couleurs
        for class_id in self.classes_of_interest:
            class_mask = mask == class_id

            # Partie gauche
            class_mask_left = np.logical_and(class_mask, mask_left)
            if np.any(class_mask_left):
                classes_present_left.append(class_id)
                pixels_left[class_id] = np.sum(class_mask_left)

                # Redimensionner l'image si nécessaire
                image_resized = cv2.resize(image, (w_mask, h_mask)) if image.shape[:2] != (h_mask, w_mask) else image

                # Extraire les couleurs dominantes basées sur la position
                if self.use_position_based:
                    n_colors = self.colors_per_class[class_id]['split']
                    left_colors[class_id], left_props[class_id] = self.extract_dominant_colors_with_position(
                        image_resized, class_mask_left, n_colors=n_colors)
                else:
                    # Méthode originale basée sur la proportion
                    n_colors = self.colors_per_class[class_id]['split']
                    left_colors[class_id], left_props[class_id] = self.extract_dominant_colors_proportion(
                        image_resized, class_mask_left, n_colors=n_colors)

            # Partie droite
            class_mask_right = np.logical_and(class_mask, mask_right)
            if np.any(class_mask_right):
                classes_present_right.append(class_id)
                pixels_right[class_id] = np.sum(class_mask_right)

                # Redimensionner l'image si nécessaire
                image_resized = cv2.resize(image, (w_mask, h_mask)) if image.shape[:2] != (h_mask, w_mask) else image

                # Extraire les couleurs dominantes basées sur la position
                if self.use_position_based:
                    n_colors = self.colors_per_class[class_id]['split']
                    right_colors[class_id], right_props[class_id] = self.extract_dominant_colors_with_position(
                        image_resized, class_mask_right, n_colors=n_colors)
                else:
                    # Méthode originale basée sur la proportion
                    n_colors = self.colors_per_class[class_id]['split']
                    right_colors[class_id], right_props[class_id] = self.extract_dominant_colors_proportion(
                        image_resized, class_mask_right, n_colors=n_colors)

        # Appliquer le lissage temporel aux couleurs si activé
        if hasattr(self, 'temporal_smoothing') and self.temporal_smoothing:
            # Lissage des couleurs à gauche
            for class_id in classes_present_left:
                if class_id in left_colors and left_colors[class_id] is not None:
                    if hasattr(self, 'previous_left_colors') and class_id in self.previous_left_colors and \
                            self.previous_left_colors[class_id] is not None:
                        # Vérifier que nous avons le même nombre de couleurs
                        current_colors = left_colors[class_id]
                        prev_colors = self.previous_left_colors[class_id]

                        # Si le nombre de couleurs est différent, ne pas faire de lissage
                        if len(current_colors) == len(prev_colors):
                            # Calculer les couleurs lissées
                            smoothed_colors = current_colors.astype(float) * self.smoothing_weight + \
                                              prev_colors.astype(float) * (1 - self.smoothing_weight)
                            left_colors[class_id] = np.clip(smoothed_colors, 0, 255).astype(int)

                    # Mettre à jour l'historique des couleurs
                    if hasattr(self, 'previous_left_colors'):
                        self.previous_left_colors[class_id] = left_colors[class_id].copy()

            # Lissage des couleurs à droite (même principe)
            for class_id in classes_present_right:
                if class_id in right_colors and right_colors[class_id] is not None:
                    if hasattr(self, 'previous_right_colors') and class_id in self.previous_right_colors and \
                            self.previous_right_colors[class_id] is not None:
                        current_colors = right_colors[class_id]
                        prev_colors = self.previous_right_colors[class_id]

                        if len(current_colors) == len(prev_colors):
                            smoothed_colors = current_colors.astype(float) * self.smoothing_weight + \
                                              prev_colors.astype(float) * (1 - self.smoothing_weight)
                            right_colors[class_id] = np.clip(smoothed_colors, 0, 255).astype(int)

                    # Mettre à jour l'historique des couleurs
                    if hasattr(self, 'previous_right_colors'):
                        self.previous_right_colors[class_id] = right_colors[class_id].copy()

        # Sauvegarder les informations de debug sur les couleurs
        if self.save_debug_images and debug_info is not None:
            debug_info_text = []
            debug_info_text.append("COULEURS DOMINANTES EXTRAITES:\n")
            debug_info_text.append("=" * 40 + "\n")

            for side, classes_present, colors_dict, props_dict in [
                ("GAUCHE", classes_present_left, left_colors, left_props),
                ("DROITE", classes_present_right, right_colors, right_props)
            ]:
                debug_info_text.append(f"\nCôté {side}:\n")
                for class_id in classes_present:
                    debug_info_text.append(f"\n  Classe {class_id} ({self.classes_of_interest[class_id]}):\n")
                    if class_id in colors_dict and colors_dict[class_id] is not None:
                        for i, (color, prop) in enumerate(zip(colors_dict[class_id], props_dict[class_id])):
                            debug_info_text.append(f"    Couleur {i + 1}: RGB{tuple(color)} - {prop * 100:.1f}%\n")

            with open(f"{self.debug_dir}/colors_{debug_info['frame_id']:04d}.txt", "w") as f:
                f.writelines(debug_info_text)

        # Si aucune classe d'intérêt n'est présente, retourner une image noire
        if not classes_present_left and not classes_present_right:
            return np.zeros((h_img, w_img, 3), dtype=np.uint8)

        # Trier les classes selon l'ordre vertical
        classes_present_left.sort(
            key=lambda x: self.vertical_order.index(x) if x in self.vertical_order else len(self.vertical_order))
        classes_present_right.sort(
            key=lambda x: self.vertical_order.index(x) if x in self.vertical_order else len(self.vertical_order))

        # Calculer les proportions relatives
        total_pixels_left = sum(pixels_left.values()) if pixels_left else 0
        total_pixels_right = sum(pixels_right.values()) if pixels_right else 0

        relative_heights_left = {class_id: (pixels_left[class_id] / total_pixels_left) * 100
                                 for class_id in classes_present_left} if total_pixels_left > 0 else {}
        relative_heights_right = {class_id: (pixels_right[class_id] / total_pixels_right) * 100
                                  for class_id in classes_present_right} if total_pixels_right > 0 else {}

        # Créer l'image de synthèse
        split_img = np.zeros((h_img, w_img, 3), dtype=np.float32)

        # Partie gauche
        if classes_present_left:
            section_heights_left = []
            remaining_height = h_img

            # Calculer les hauteurs des sections
            for i, class_id in enumerate(classes_present_left):
                if i == len(classes_present_left) - 1:
                    section_heights_left.append(remaining_height)
                else:
                    section_height = int((relative_heights_left[class_id] / 100) * h_img)
                    section_height = max(self.min_band_height if hasattr(self, 'min_band_height') else 3,
                                         section_height)  # Garantir une hauteur minimale
                    section_heights_left.append(section_height)
                    remaining_height -= section_height

            y_pos_left = 0
            for i, class_id in enumerate(classes_present_left):
                section_height = section_heights_left[i]

                if class_id in left_colors and left_colors[class_id] is not None:
                    colors = left_colors[class_id]
                    props = left_props[class_id]

                    # Normaliser les proportions
                    total_prop = sum(props)
                    if total_prop > 0:
                        normalized_props = props / total_prop

                        # Calculer les hauteurs de chaque couleur
                        color_heights = []
                        remaining = section_height

                        for j, prop in enumerate(normalized_props):
                            if j == len(normalized_props) - 1:
                                color_heights.append(remaining)
                            else:
                                color_height = int(prop * section_height)
                                color_height = max(self.min_band_height if hasattr(self, 'min_band_height') else 3,
                                                   color_height)  # Hauteur minimale pour chaque bande
                                color_heights.append(color_height)
                                remaining -= color_height

                        # Dessiner avec ou sans interpolation
                        section_y = y_pos_left
                        for j, (color, height) in enumerate(zip(colors, color_heights)):
                            if height > 0 and section_y < h_img:
                                if self.use_interpolation:
                                    # Interpolation avec la couleur suivante ou la prochaine classe
                                    if j == len(colors) - 1 and i < len(classes_present_left) - 1:
                                        next_class_id = classes_present_left[i + 1]
                                        if next_class_id in left_colors and left_colors[next_class_id] is not None:
                                            next_color = left_colors[next_class_id][0]
                                            # Calculer la taille d'interpolation
                                            interp_size = max(self.interpolation_strength, height // 3)
                                            interp_size = min(interp_size,
                                                              height - 1)  # Garder au moins 1 pixel non interpolé

                                            # Dessiner la partie non interpolée
                                            if height - interp_size > 0:
                                                split_img[section_y:section_y + height - interp_size,
                                                :mid_w_img] = color

                                            # Créer l'interpolation entre classes
                                            if interp_size > 0:
                                                gradient = self.interpolate_colors(color, next_color, interp_size)
                                                for k in range(interp_size):
                                                    if section_y + height - interp_size + k < h_img:
                                                        split_img[section_y + height - interp_size + k, :mid_w_img] = \
                                                        gradient[k]
                                        else:
                                            end_y = min(section_y + height, h_img)
                                            split_img[section_y:end_y, :mid_w_img] = color
                                    # Si ce n'est pas la dernière couleur, interpoler avec la suivante
                                    elif j < len(colors) - 1:
                                        next_color = colors[j + 1]
                                        # Calculer la taille d'interpolation
                                        interp_size = max(self.interpolation_strength, height // 3)
                                        interp_size = min(interp_size,
                                                          height - 1)  # Garder au moins 1 pixel non interpolé

                                        # Dessiner la partie non interpolée
                                        if height - interp_size > 0:
                                            split_img[section_y:section_y + height - interp_size, :mid_w_img] = color

                                        # Créer l'interpolation entre couleurs
                                        if interp_size > 0:
                                            gradient = self.interpolate_colors(color, next_color, interp_size)
                                            for k in range(interp_size):
                                                if section_y + height - interp_size + k < h_img:
                                                    split_img[section_y + height - interp_size + k, :mid_w_img] = \
                                                    gradient[k]
                                    else:
                                        end_y = min(section_y + height, h_img)
                                        split_img[section_y:end_y, :mid_w_img] = color
                                else:
                                    # Sans interpolation
                                    end_y = min(section_y + height, h_img)
                                    split_img[section_y:end_y, :mid_w_img] = color

                            section_y += height

                y_pos_left += section_height

        # Partie droite (similaire à la partie gauche mais pour le côté droit)
        if classes_present_right:
            section_heights_right = []
            remaining_height = h_img

            for i, class_id in enumerate(classes_present_right):
                if i == len(classes_present_right) - 1:
                    section_heights_right.append(remaining_height)
                else:
                    section_height = int((relative_heights_right[class_id] / 100) * h_img)
                    section_height = max(self.min_band_height if hasattr(self, 'min_band_height') else 3,
                                         section_height)  # Garantir une hauteur minimale
                    section_heights_right.append(section_height)
                    remaining_height -= section_height

            y_pos_right = 0
            for i, class_id in enumerate(classes_present_right):
                section_height = section_heights_right[i]

                if class_id in right_colors and right_colors[class_id] is not None:
                    colors = right_colors[class_id]
                    props = right_props[class_id]

                    # Normaliser les proportions
                    total_prop = sum(props)
                    if total_prop > 0:
                        normalized_props = props / total_prop

                        # Calculer les hauteurs de chaque couleur
                        color_heights = []
                        remaining = section_height

                        for j, prop in enumerate(normalized_props):
                            if j == len(normalized_props) - 1:
                                color_heights.append(remaining)
                            else:
                                color_height = int(prop * section_height)
                                color_height = max(self.min_band_height if hasattr(self, 'min_band_height') else 3,
                                                   color_height)  # Hauteur minimale pour chaque bande
                                color_heights.append(color_height)
                                remaining -= color_height

                        # Dessiner avec ou sans interpolation
                        section_y = y_pos_right
                        for j, (color, height) in enumerate(zip(colors, color_heights)):
                            if height > 0 and section_y < h_img:
                                if self.use_interpolation:
                                    # Interpolation avec la couleur suivante ou la prochaine classe
                                    if j == len(colors) - 1 and i < len(classes_present_right) - 1:
                                        next_class_id = classes_present_right[i + 1]
                                        if next_class_id in right_colors and right_colors[next_class_id] is not None:
                                            next_color = right_colors[next_class_id][0]
                                            # Calculer la taille d'interpolation
                                            interp_size = max(self.interpolation_strength, height // 3)
                                            interp_size = min(interp_size,
                                                              height - 1)  # Garder au moins 1 pixel non interpolé

                                            # Dessiner la partie non interpolée
                                            if height - interp_size > 0:
                                                split_img[section_y:section_y + height - interp_size,
                                                mid_w_img:] = color

                                            # Créer l'interpolation entre classes
                                            if interp_size > 0:
                                                gradient = self.interpolate_colors(color, next_color, interp_size)
                                                for k in range(interp_size):
                                                    if section_y + height - interp_size + k < h_img:
                                                        split_img[section_y + height - interp_size + k, mid_w_img:] = \
                                                        gradient[k]
                                        else:
                                            end_y = min(section_y + height, h_img)
                                            split_img[section_y:end_y, mid_w_img:] = color
                                    # Si ce n'est pas la dernière couleur, interpoler avec la suivante
                                    elif j < len(colors) - 1:
                                        next_color = colors[j + 1]
                                        # Calculer la taille d'interpolation
                                        interp_size = max(self.interpolation_strength, height // 3)
                                        interp_size = min(interp_size,
                                                          height - 1)  # Garder au moins 1 pixel non interpolé

                                        # Dessiner la partie non interpolée
                                        if height - interp_size > 0:
                                            split_img[section_y:section_y + height - interp_size, mid_w_img:] = color

                                        # Créer l'interpolation entre couleurs
                                        if interp_size > 0:
                                            gradient = self.interpolate_colors(color, next_color, interp_size)
                                            for k in range(interp_size):
                                                if section_y + height - interp_size + k < h_img:
                                                    split_img[section_y + height - interp_size + k, mid_w_img:] = \
                                                    gradient[k]
                                    else:
                                        end_y = min(section_y + height, h_img)
                                        split_img[section_y:end_y, mid_w_img:] = color
                                else:
                                    # Sans interpolation
                                    end_y = min(section_y + height, h_img)
                                    split_img[section_y:end_y, mid_w_img:] = color

                            section_y += height

                y_pos_right += section_height

        # Convertir en uint8 pour l'affichage
        split_img = np.clip(split_img, 0, 255).astype(np.uint8)

        return split_img

    def extract_dominant_colors_proportion(self, image, boolean_mask, n_colors=3):
        """
        Version de l'extraction des couleurs dominantes basée sur la proportion
        """
        # Si aucun pixel n'est sélectionné, retourner None
        if not np.any(boolean_mask):
            return None, None

        # Extraire les pixels correspondants
        pixels = image[boolean_mask]

        # Sous-échantillonnage pour accélérer le clustering
        if len(pixels) > 10000:  # Si beaucoup de pixels
            sample_size = min(int(10000 * self.sample_size_factor), len(pixels) // 2)
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            pixels = pixels[indices]

        # Éviter une erreur si le nombre de pixels est inférieur à n_colors
        if len(pixels) < n_colors:
            n_colors = max(1, len(pixels))

        # Appliquer KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42,
                        n_init=self.kmeans_inits, max_iter=self.kmeans_max_iter)
        kmeans.fit(pixels)

        # Obtenir les couleurs et leur proportion
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_

        # Calculer les proportions
        proportions = np.bincount(labels) / len(labels)

        # Trier par proportion
        idx = np.argsort(-proportions)
        colors = colors[idx]
        proportions = proportions[idx]

        return colors, proportions

    def process_frame(self, frame, frame_id=0):
        """Traite une frame pour générer l'image split"""
        # Prétraiter et segmenter l'image
        input_tensor = self.preprocess_image(frame).to(self.device)

        # Inférence avec le modèle
        with torch.no_grad():
            output = self.model(input_tensor)

            if isinstance(output, dict) and 'out' in output:
                output_tensor = output['out'][0]
            elif isinstance(output, torch.Tensor):
                output_tensor = output[0]
            else:
                raise ValueError(f"Structure de sortie inattendue: {type(output)}")

        # Récupérer le masque de segmentation
        mask = output_tensor.argmax(0).cpu().numpy()

        # Debug info pour sauvegarder des images intermédiaires
        debug_info = {'frame_id': frame_id} if self.save_debug_images else None

        # Sauvegarder l'image originale si debug est activé
        if self.save_debug_images:
            # Convertir en BGR pour OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{self.debug_dir}/original_{frame_id:04d}.jpg", frame_bgr)

        # Créer l'image split
        current_split_img = self.create_split_image(frame, mask, debug_info)

        # Appliquer le lissage temporel si activé et s'il y a des frames précédentes
        if self.temporal_smoothing and self.previous_frames:
            # Calculer la moyenne pondérée des images
            smoothed_img = current_split_img.astype(float) * self.smoothing_weight

            # Calculer la luminance moyenne avant lissage
            original_luminance = np.mean(current_split_img)

            # Ajouter les frames précédentes avec poids décroissant
            remaining_weight = 1.0 - self.smoothing_weight
            for i, prev_frame in enumerate(reversed(self.previous_frames)):
                if i >= self.smoothing_frames - 1:
                    break  # Limiter au nombre de frames spécifié

                # Poids exponentiel décroissant
                weight = remaining_weight * (0.7 ** i)
                smoothed_img += prev_frame.astype(float) * weight

            # Calculer la luminance après lissage
            smoothed_luminance = np.mean(smoothed_img)

            # Correction simple de luminance
            if smoothed_luminance > 0:  # Éviter division par zéro
                correction_factor = original_luminance / smoothed_luminance
                smoothed_img *= correction_factor

            # Convertir en uint8 pour l'affichage
            split_img = np.clip(smoothed_img, 0, 255).astype(np.uint8)
        else:
            # Utiliser l'image courante sans lissage
            split_img = current_split_img

        # Mettre à jour l'historique des frames
        self.previous_frames.append(current_split_img)
        if len(self.previous_frames) > self.smoothing_frames:
            self.previous_frames.pop(0)  # Supprimer la plus ancienne frame

        # Sauvegarder l'image de sortie si debug est activé
        if self.save_debug_images:
            # Convertir en BGR pour OpenCV
            split_bgr = cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{self.debug_dir}/split_{frame_id:04d}.jpg", split_bgr)

        return split_img

    def process_video(self, video_path, output_path=None):
        """
        Traite une vidéo et génère la version synthétisée avec mesure précise des performances

        Args:
            video_path: Chemin vers la vidéo d'entrée
            output_path: Chemin pour enregistrer la vidéo de sortie (None pour ne pas enregistrer)
        """
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
            return

        # Obtenir les propriétés de la vidéo
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Vidéo: {frame_width}x{frame_height}, {fps_video} FPS, {total_frames} frames")

        # Configurer l'enregistreur vidéo si nécessaire
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps_video, (frame_width, frame_height))

        # Variables pour le calcul de performance
        start_time_total = time.time()
        frame_times = []
        frame_count = 0

        try:
            while True:
                # Mesurer le temps de traitement
                start_time = time.time()

                # Lire une frame de la vidéo
                ret, frame = cap.read()
                if not ret:
                    break

                # Afficher la progression
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Traitement: {frame_count}/{total_frames} frames ({frame_count / total_frames * 100:.1f}%)")

                # Convertir la frame de BGR à RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Traiter la frame
                split_img = self.process_frame(frame_rgb, frame_count)

                # Convertir l'image de synthèse en BGR pour OpenCV
                output_img = cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR)

                # Enregistrer la vidéo de sortie si demandé
                if video_writer:
                    video_writer.write(output_img)

                # Calculer et stocker le temps de traitement
                frame_time = time.time() - start_time
                frame_times.append(frame_time)

                # Afficher le FPS actuel
                if frame_count % 100 == 0:
                    current_fps = 1 / frame_time
                    print(f"Frame {frame_count}: {frame_time * 1000:.1f} ms ({current_fps:.1f} FPS)")

        finally:
            # Libérer les ressources
            cap.release()
            if video_writer:
                video_writer.release()

            # Calculer et afficher les statistiques de performance
            end_time_total = time.time()
            total_processing_time = end_time_total - start_time_total

            if frame_count > 0:
                avg_fps = frame_count / total_processing_time
                avg_frame_time = sum(frame_times) / len(frame_times)
                max_frame_time = max(frame_times)
                min_frame_time = min(frame_times)

                min_fps = 1.0 / max_frame_time if max_frame_time > 0 else float('inf')
                max_fps = 1.0 / min_frame_time if min_frame_time > 0 else float('inf')

                print("\n==== STATISTIQUES DE PERFORMANCE ====")
                print(f"Frames traitées: {frame_count}/{total_frames}")
                print(f"Temps total: {total_processing_time:.2f} secondes")
                print(f"FPS moyen: {avg_fps:.2f}")
                print(f"Temps moyen par frame: {avg_frame_time * 1000:.2f} ms")
                print(f"Temps max par frame: {max_frame_time * 1000:.2f} ms (min FPS: {min_fps:.2f})")
                print(f"Temps min par frame: {min_frame_time * 1000:.2f} ms (max FPS: {max_fps:.2f})")

                # Évaluation pour application temps réel
                if avg_fps >= 30:
                    print("\nPerformance: EXCELLENTE pour temps réel (>30 FPS en moyenne)")
                elif avg_fps >= 24:
                    print("\nPerformance: BONNE pour temps réel (>24 FPS en moyenne)")
                elif avg_fps >= 15:
                    print("\nPerformance: ACCEPTABLE pour temps réel avec latence visible")
                else:
                    print("\nPerformance: INSUFFISANTE pour temps réel fluide")

                if min_fps < 15:
                    print(f"ATTENTION: Chutes de performance détectées (minimum {min_fps:.1f} FPS)")

            if output_path:
                print(f"Vidéo de sortie enregistrée dans: {output_path}")

    def run_from_camera(self):
        """Exécute le processeur à partir d'une caméra avec mesure de performance"""
        # Ouvrir la caméra
        camera_index = 0  # Utiliser la caméra par défaut (index 0)
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Erreur: Impossible d'ouvrir la caméra {camera_index}")
            return

        # Configurer la caméra pour de meilleures performances
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Afficher les propriétés de la caméra
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Caméra configurée: {actual_width}x{actual_height}, {actual_fps} FPS")

        print("Démarrage du traitement vidéo en temps réel")
        print("Appuyez sur 'q' pour quitter, 's' pour prendre une capture d'écran")

        # Variables pour le calcul de performance
        start_time_total = time.time()
        frame_times = []
        frame_count = 0
        fps_window = []  # Pour calculer le FPS glissant
        fps_window_size = 30  # FPS moyen sur 30 frames

        try:
            while True:
                # Mesurer le temps de traitement
                start_time = time.time()

                # Lire une frame de la caméra
                ret, frame = cap.read()
                if not ret:
                    print("Erreur: Impossible de lire la frame de la caméra")
                    break

                # Convertir la frame de BGR à RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Traiter la frame
                split_img = self.process_frame(frame_rgb, frame_count)
                frame_count += 1

                # Convertir l'image de synthèse en BGR pour OpenCV
                output_img = cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR)

                # Calculer le FPS
                frame_time = time.time() - start_time
                frame_times.append(frame_time)
                fps_window.append(1 / frame_time)
                if len(fps_window) > fps_window_size:
                    fps_window.pop(0)

                # FPS glissant
                current_fps = sum(fps_window) / len(fps_window)

                # Afficher le FPS sur la sortie (optionnel, peut être commenté pour optimisation)
                if frame_count % 30 == 0:
                    print(f"FPS actuel: {current_fps:.1f}, Temps frame: {frame_time * 1000:.1f} ms")

                # Afficher les images
                cv2.imshow('Sortie Split', output_img)

                # Sauvegarder une capture si demandé
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Traitement interrompu par l'utilisateur.")
                    break
                elif key == ord('s'):
                    # Prendre une capture d'écran
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    capture_path = f"capture_{timestamp}.jpg"
                    cv2.imwrite(capture_path, output_img)
                    print(f"Capture d'écran sauvegardée: {capture_path}")

                # Simulation de l'envoi OSC (décommentez si vous voulez tester)
                # band_colors = self.simulate_osc_transmission(split_img)

        finally:
            # Libérer les ressources
            cap.release()
            cv2.destroyAllWindows()

            # Calculer et afficher les statistiques de performance
            end_time_total = time.time()
            total_processing_time = end_time_total - start_time_total

            if frame_count > 0:
                avg_fps = frame_count / total_processing_time
                avg_frame_time = sum(frame_times) / len(frame_times)
                max_frame_time = max(frame_times)
                min_frame_time = min(frame_times)

                min_fps = 1.0 / max_frame_time if max_frame_time > 0 else float('inf')
                max_fps = 1.0 / min_frame_time if min_frame_time > 0 else float('inf')

                print("\n==== STATISTIQUES DE PERFORMANCE ====")
                print(f"Durée du test: {total_processing_time:.2f} secondes")
                print(f"Frames traitées: {frame_count}")
                print(f"FPS moyen: {avg_fps:.2f}")
                print(f"Temps moyen par frame: {avg_frame_time * 1000:.2f} ms")
                print(f"Temps max par frame: {max_frame_time * 1000:.2f} ms (min FPS: {min_fps:.2f})")
                print(f"Temps min par frame: {min_frame_time * 1000:.2f} ms (max FPS: {max_fps:.2f})")

                # Évaluation pour application temps réel
                if avg_fps >= 30:
                    print("\nPerformance: EXCELLENTE pour temps réel (>30 FPS en moyenne)")
                elif avg_fps >= 24:
                    print("\nPerformance: BONNE pour temps réel (>24 FPS en moyenne)")
                elif avg_fps >= 15:
                    print("\nPerformance: ACCEPTABLE pour temps réel avec latence visible")
                else:
                    print("\nPerformance: INSUFFISANTE pour temps réel fluide")

                if min_fps < 15:
                    print(f"ATTENTION: Chutes de performance détectées (minimum {min_fps:.1f} FPS)")

# Si ce script est exécuté directement
if __name__ == "__main__":
    # Créer le synthétiseur
    synthesizer = VideoSplitSynthesizer()

    # Choix du mode
    print("Choisissez le mode de fonctionnement:")
    print("1 - Tester avec une vidéo")
    print("2 - Utiliser la caméra")
    print("3 - Traiter une image statique")

    choice = input("Votre choix (1, 2 ou 3): ")

    video_inputs_dir = os.path.join(base_dir, "video_inputs")
    video_outputs_dir = os.path.join(base_dir, "video_outputs")
    os.makedirs(video_outputs_dir, exist_ok=True)

    if choice == "1":
        # Mode vidéo
        video_name = input("Nom de la vidéo à traiter : ")
        video_path = os.path.join(video_inputs_dir, video_name)
        save_output = input("Voulez-vous enregistrer la vidéo de sortie? (o/n): ")

        output_path = None
        if save_output.lower() == "o" or save_output.lower() == "oui":
            output_path = os.path.join(video_outputs_dir, video_name)

        # Options de qualité
        print("\nChoisissez le préréglage de qualité:")
        print("1 - Performance (plus rapide, qualité moindre)")
        print("2 - Équilibré (recommandé)")
        print("3 - Qualité (plus lent, meilleure qualité)")
        quality_choice = input("Votre choix (1, 2 ou 3): ")

        if quality_choice == "1":
            synthesizer.quality_preset = "performance"
        elif quality_choice == "3":
            synthesizer.quality_preset = "quality"
        else:
            synthesizer.quality_preset = "balanced"

        synthesizer.update_quality_settings()

        # Rappeler les options spéciales
        print("\nPendant l'exécution, vous pouvez utiliser:")
        print("  'q' pour quitter")
        print("  '1', '2', '3' pour changer le préréglage de qualité")
        print("  'i' pour activer/désactiver l'interpolation")
        print("  '+' et '-' pour ajuster la force d'interpolation")
        print("  'p' pour activer/désactiver le positionnement basé sur l'image")

        # Traiter la vidéo
        synthesizer.process_video(video_path, output_path)

    elif choice == "2":
        # Mode caméra
        print("\nOptions de la caméra:")
        print("1 - Performance (recommandé pour webcams/ordinateurs moins puissants)")
        print("2 - Équilibré")
        print("3 - Qualité (ordinateurs puissants uniquement)")
        quality_choice = input("Votre choix (1, 2 ou 3): ")

        if quality_choice == "1":
            synthesizer.quality_preset = "performance"
        elif quality_choice == "3":
            synthesizer.quality_preset = "quality"
        else:
            synthesizer.quality_preset = "balanced"

        synthesizer.update_quality_settings()

        # Rappeler les options spéciales
        print("\nPendant l'exécution, vous pouvez utiliser:")
        print("  'q' pour quitter")
        print("  '1', '2', '3' pour changer le préréglage de qualité")
        print("  'i' pour activer/désactiver l'interpolation")
        print("  'p' pour activer/désactiver le positionnement basé sur l'image")
        print("  '+' et '-' pour ajuster la force d'interpolation")
        print("  's' pour prendre une capture d'écran")

        # Mode caméra
        synthesizer.run_from_camera()

    elif choice == "3":
        # Mode image statique
        image_path = input("Chemin vers l'image à traiter: ")

        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erreur: Impossible de charger l'image {image_path}")
            exit()

        # Convertir l'image en RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Réglage de la qualité maximale pour l'image statique
        synthesizer.quality_preset = "quality"
        synthesizer.update_quality_settings()

        # Traiter l'image
        split_img = synthesizer.process_frame(image_rgb, 1)

        # Convertir l'image de synthèse en BGR pour OpenCV
        output_img = cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR)

        # Sauvegarder l'image
        output_path = f"synthese_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, output_img)
        print(f"Image synthétisée sauvegardée dans: {output_path}")

        # Afficher les images
        # cv2.imshow('Image originale', image)
        # cv2.imshow('Synthèse split', output_img)

        print("Appuyez sur une touche pour quitter...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Choix non valide.")

