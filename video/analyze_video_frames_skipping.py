"""
================================================================================
 📄 Script : analyze_video_frames_skipping.py
 📦 Projet : New Chameleon Strategy – Partie Vidéo (V2)
 🧠 Objectif :
     Ce script applique une segmentation sémantique DeepLabV3+ frame par frame
     sur une vidéo locale, mais en ne traitant qu'une frame sur un intervalle défini
     (par exemple 1 sur 31) et en générant les frames intermédiaires par interpolation
     pour obtenir une animation fluide.

     Ceci permet de réduire le temps de traitement (en évitant d'exécuter l'inférence
     coûteuse sur chaque frame) tout en conservant une fluidité visuelle.

 🧰 Fonctionnalités :
     - Chargement du modèle DeepLabV3+ (MobileNet) depuis checkpoints/
     - Lecture d'une vidéo depuis le dossier video_inputs/
     - Traitement uniquement d'une frame sur un ensemble (exemple : 1 sur 31)
     - Interpolation linéaire pour générer les frames intermédiaires entre deux frames traitées
     - Génération d'une vidéo en sortie avec les masques segmentés colorés selon la palette Cityscapes
     - Affichage des statistiques de performance
     - Création automatique d'un dossier video_outputs/

 📎 Répertoires utilisés :
     - video_inputs/ : contient les vidéos d'entrée à traiter
     - video_outputs/ : vidéos générées automatiquement (non versionnées)

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
        """Initialisation du synthétiseur d'images split (version skipping)"""

        # Sélection du device
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

        # Définir les classes d'intérêt et l'ordre vertical
        self.classes_of_interest = {
            8: 'végétation',
            9: 'terrain',
            10: 'ciel'
        }
        self.vertical_order = [10, 8, 9]  # [ciel, végétation, terrain]
        self.colors_per_class = {
            8: {'split': 3},  # végétation: 3 couleurs par côté
            9: {'split': 3},  # terrain: 3 couleurs par côté
            10: {'split': 2},  # ciel: 2 couleurs par côté
        }

        # Paramètres d'optimisation
        self.use_interpolation = True
        self.interpolation_strength = 10
        self.use_position_based = True

        # Préréglage de qualité
        self.quality_preset = "balanced"  # options: "performance", "balanced", "quality"
        self.update_quality_settings()

        # Paramètres du lissage temporel
        self.temporal_smoothing = True
        self.smoothing_frames = 5
        self.smoothing_weight = 0.6
        self.previous_frames = []
        self.previous_left_colors = {}
        self.previous_right_colors = {}

    def update_quality_settings(self):
        """Met à jour les paramètres en fonction du préréglage de qualité choisi"""
        if self.quality_preset == "performance":
            self.processing_resolution = (384, 384)
            self.sample_size_factor = 0.25
            self.kmeans_inits = 1
            self.kmeans_max_iter = 100
            self.min_band_height = 3
        elif self.quality_preset == "balanced":
            self.processing_resolution = (448, 448)
            self.sample_size_factor = 0.4
            self.kmeans_inits = 2
            self.kmeans_max_iter = 150
            self.min_band_height = 4
        else:  # "quality"
            self.processing_resolution = (512, 512)
            self.sample_size_factor = 0.6
            self.kmeans_inits = 3
            self.kmeans_max_iter = 200
            self.min_band_height = 5

        print(f"Qualité: {self.quality_preset}, Résolution: {self.processing_resolution}")

    def preprocess_image(self, image):
        """Prétraite l'image pour le modèle"""
        resized = cv2.resize(image, self.processing_resolution)
        input_tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()
        input_tensor = input_tensor.div(255.0)
        input_tensor = input_tensor.sub(torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1))
        input_tensor = input_tensor.div(torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1))
        return input_tensor.unsqueeze(0)

    def extract_dominant_colors_with_position(self, image, boolean_mask, n_colors=3):
        """Extraire les couleurs dominantes en gardant l'information de position verticale"""
        if not np.any(boolean_mask):
            return None, None

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

        # Sous-échantillonnage pour accélérer le clustering
        if len(pixels) > 10000:
            sample_size = min(int(10000 * self.sample_size_factor), len(pixels) // 2)
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            pixels = pixels[indices]
            positions_y = positions_y[indices]

        # Si le nombre de pixels est insuffisant, ajuster n_colors
        if len(pixels) < n_colors:
            n_colors = max(1, len(pixels))

        # Vérifications pour éviter les avertissements KMeans
        if pixels.size == 0:
            return None, None
        if np.allclose(pixels, pixels[0], atol=1e-6):
            color = pixels[0].astype(int)
            return np.array([color]), np.array([1.0])
        if np.all(np.std(pixels, axis=0) < 1e-3):
            color = np.mean(pixels, axis=0).astype(int)
            return np.array([color]), np.array([1.0])

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42,
                        n_init=self.kmeans_inits, max_iter=self.kmeans_max_iter)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        proportions = np.bincount(labels) / len(labels)

        # Calcul de la position moyenne pour chaque cluster
        cluster_positions = []
        for i in range(n_colors):
            cluster_mask = labels == i
            if np.any(cluster_mask):
                avg_pos = np.median(positions_y[cluster_mask])
                cluster_positions.append(avg_pos)
            else:
                cluster_positions.append(0)
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

    def create_split_image(self, image, mask):
        """Crée l'image split à partir de l'image et du masque de segmentation"""
        h_img, w_img = image.shape[:2]
        h_mask, w_mask = mask.shape

        mid_w_img = w_img // 2
        mid_w_mask = w_mask // 2

        mask_left = np.zeros_like(mask, dtype=bool)
        mask_right = np.zeros_like(mask, dtype=bool)
        mask_left[:, :mid_w_mask] = True
        mask_right[:, mid_w_mask:] = True

        pixels_left = {}
        pixels_right = {}
        classes_present_left = []
        classes_present_right = []
        left_colors = {}
        right_colors = {}
        left_props = {}
        right_props = {}

        for class_id in self.classes_of_interest:
            class_mask = mask == class_id

            # Partie gauche
            class_mask_left = np.logical_and(class_mask, mask_left)
            if np.any(class_mask_left):
                classes_present_left.append(class_id)
                pixels_left[class_id] = np.sum(class_mask_left)
                image_resized = cv2.resize(image, (w_mask, h_mask)) if image.shape[:2] != (h_mask, w_mask) else image
                n_colors = self.colors_per_class[class_id]['split']
                left_colors[class_id], left_props[class_id] = self.extract_dominant_colors_with_position(
                    image_resized, class_mask_left, n_colors=n_colors)

            # Partie droite
            class_mask_right = np.logical_and(class_mask, mask_right)
            if np.any(class_mask_right):
                classes_present_right.append(class_id)
                pixels_right[class_id] = np.sum(class_mask_right)
                image_resized = cv2.resize(image, (w_mask, h_mask)) if image.shape[:2] != (h_mask, w_mask) else image
                n_colors = self.colors_per_class[class_id]['split']
                right_colors[class_id], right_props[class_id] = self.extract_dominant_colors_with_position(
                    image_resized, class_mask_right, n_colors=n_colors)

        # Lissage temporel des couleurs
        if self.temporal_smoothing:
            for class_id in classes_present_left:
                if class_id in left_colors and left_colors[class_id] is not None:
                    if class_id in self.previous_left_colors and self.previous_left_colors[class_id] is not None:
                        current_colors = left_colors[class_id]
                        prev_colors = self.previous_left_colors[class_id]
                        if len(current_colors) == len(prev_colors):
                            smoothed_colors = current_colors.astype(float) * self.smoothing_weight + \
                                              prev_colors.astype(float) * (1 - self.smoothing_weight)
                            left_colors[class_id] = np.clip(smoothed_colors, 0, 255).astype(int)
                    self.previous_left_colors[class_id] = left_colors[class_id].copy()
            for class_id in classes_present_right:
                if class_id in right_colors and right_colors[class_id] is not None:
                    if class_id in self.previous_right_colors and self.previous_right_colors[class_id] is not None:
                        current_colors = right_colors[class_id]
                        prev_colors = self.previous_right_colors[class_id]
                        if len(current_colors) == len(prev_colors):
                            smoothed_colors = current_colors.astype(float) * self.smoothing_weight + \
                                              prev_colors.astype(float) * (1 - self.smoothing_weight)
                            right_colors[class_id] = np.clip(smoothed_colors, 0, 255).astype(int)
                    self.previous_right_colors[class_id] = right_colors[class_id].copy()

        if not classes_present_left and not classes_present_right:
            return np.zeros((h_img, w_img, 3), dtype=np.uint8)

        classes_present_left.sort(
            key=lambda x: self.vertical_order.index(x) if x in self.vertical_order else len(self.vertical_order))
        classes_present_right.sort(
            key=lambda x: self.vertical_order.index(x) if x in self.vertical_order else len(self.vertical_order))

        total_pixels_left = sum(pixels_left.values()) if pixels_left else 0
        total_pixels_right = sum(pixels_right.values()) if pixels_right else 0

        relative_heights_left = {class_id: (pixels_left[class_id] / total_pixels_left) * 100
                                 for class_id in classes_present_left} if total_pixels_left > 0 else {}
        relative_heights_right = {class_id: (pixels_right[class_id] / total_pixels_right) * 100
                                  for class_id in classes_present_right} if total_pixels_right > 0 else {}

        split_img = np.zeros((h_img, w_img, 3), dtype=np.float32)

        # Partie gauche
        if classes_present_left:
            section_heights_left = []
            remaining_height = h_img
            for i, class_id in enumerate(classes_present_left):
                if i == len(classes_present_left) - 1:
                    section_heights_left.append(remaining_height)
                else:
                    section_height = int((relative_heights_left[class_id] / 100) * h_img)
                    section_height = max(self.min_band_height, section_height)
                    section_heights_left.append(section_height)
                    remaining_height -= section_height

            y_pos_left = 0
            for i, class_id in enumerate(classes_present_left):
                section_height = section_heights_left[i]
                if class_id in left_colors and left_colors[class_id] is not None:
                    colors = left_colors[class_id]
                    props = left_props[class_id]
                    total_prop = sum(props)
                    if total_prop > 0:
                        normalized_props = props / total_prop
                        color_heights = []
                        remaining = section_height
                        for j, prop in enumerate(normalized_props):
                            if j == len(normalized_props) - 1:
                                color_heights.append(remaining)
                            else:
                                color_height = int(prop * section_height)
                                color_height = max(self.min_band_height, color_height)
                                color_heights.append(color_height)
                                remaining -= color_height

                        section_y = y_pos_left
                        for j, (color, height) in enumerate(zip(colors, color_heights)):
                            if height > 0 and section_y < h_img:
                                if self.use_interpolation:
                                    if j == len(colors) - 1 and i < len(classes_present_left) - 1:
                                        next_class_id = classes_present_left[i + 1]
                                        if next_class_id in left_colors and left_colors[next_class_id] is not None:
                                            next_color = left_colors[next_class_id][0]
                                            interp_size = max(self.interpolation_strength, height // 3)
                                            interp_size = min(interp_size, height - 1)
                                            if height - interp_size > 0:
                                                split_img[section_y:section_y + height - interp_size,
                                                :mid_w_img] = color
                                            if interp_size > 0:
                                                gradient = self.interpolate_colors(color, next_color, interp_size)
                                                for k in range(interp_size):
                                                    if section_y + height - interp_size + k < h_img:
                                                        split_img[section_y + height - interp_size + k, :mid_w_img] = \
                                                        gradient[k]
                                        else:
                                            end_y = min(section_y + height, h_img)
                                            split_img[section_y:end_y, :mid_w_img] = color
                                    elif j < len(colors) - 1:
                                        next_color = colors[j + 1]
                                        interp_size = max(self.interpolation_strength, height // 3)
                                        interp_size = min(interp_size, height - 1)
                                        if height - interp_size > 0:
                                            split_img[section_y:section_y + height - interp_size, :mid_w_img] = color
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
                                    end_y = min(section_y + height, h_img)
                                    split_img[section_y:end_y, :mid_w_img] = color
                            section_y += height
                y_pos_left += section_height

        # Partie droite
        if classes_present_right:
            section_heights_right = []
            remaining_height = h_img
            for i, class_id in enumerate(classes_present_right):
                if i == len(classes_present_right) - 1:
                    section_heights_right.append(remaining_height)
                else:
                    section_height = int((relative_heights_right[class_id] / 100) * h_img)
                    section_height = max(self.min_band_height, section_height)
                    section_heights_right.append(section_height)
                    remaining_height -= section_height

            y_pos_right = 0
            for i, class_id in enumerate(classes_present_right):
                section_height = section_heights_right[i]
                if class_id in right_colors and right_colors[class_id] is not None:
                    colors = right_colors[class_id]
                    props = right_props[class_id]
                    total_prop = sum(props)
                    if total_prop > 0:
                        normalized_props = props / total_prop
                        color_heights = []
                        remaining = section_height
                        for j, prop in enumerate(normalized_props):
                            if j == len(normalized_props) - 1:
                                color_heights.append(remaining)
                            else:
                                color_height = int(prop * section_height)
                                color_height = max(self.min_band_height, color_height)
                                color_heights.append(color_height)
                                remaining -= color_height

                        section_y = y_pos_right
                        for j, (color, height) in enumerate(zip(colors, color_heights)):
                            if height > 0 and section_y < h_img:
                                if self.use_interpolation:
                                    if j == len(colors) - 1 and i < len(classes_present_right) - 1:
                                        next_class_id = classes_present_right[i + 1]
                                        if next_class_id in right_colors and right_colors[next_class_id] is not None:
                                            next_color = right_colors[next_class_id][0]
                                            interp_size = max(self.interpolation_strength, height // 3)
                                            interp_size = min(interp_size, height - 1)
                                            if height - interp_size > 0:
                                                split_img[section_y:section_y + height - interp_size,
                                                mid_w_img:] = color
                                            if interp_size > 0:
                                                gradient = self.interpolate_colors(color, next_color, interp_size)
                                                for k in range(interp_size):
                                                    if section_y + height - interp_size + k < h_img:
                                                        split_img[section_y + height - interp_size + k, mid_w_img:] = \
                                                        gradient[k]
                                        else:
                                            end_y = min(section_y + height, h_img)
                                            split_img[section_y:end_y, mid_w_img:] = color
                                    elif j < len(colors) - 1:
                                        next_color = colors[j + 1]
                                        interp_size = max(self.interpolation_strength, height // 3)
                                        interp_size = min(interp_size, height - 1)
                                        if height - interp_size > 0:
                                            split_img[section_y:section_y + height - interp_size, mid_w_img:] = color
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
                                    end_y = min(section_y + height, h_img)
                                    split_img[section_y:end_y, mid_w_img:] = color
                            section_y += height
                y_pos_right += section_height

        split_img = np.clip(split_img, 0, 255).astype(np.uint8)
        return split_img

    def process_frame(self, frame):
        """Traite une frame pour générer l'image split"""
        input_tensor = self.preprocess_image(frame).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, dict) and 'out' in output:
                output_tensor = output['out'][0]
            elif isinstance(output, torch.Tensor):
                output_tensor = output[0]
            else:
                raise ValueError(f"Structure de sortie inattendue: {type(output)}")
        mask = output_tensor.argmax(0).cpu().numpy()
        current_split_img = self.create_split_image(frame, mask)

        # Lissage temporel des frames
        if self.temporal_smoothing and self.previous_frames:
            smoothed_img = current_split_img.astype(float) * self.smoothing_weight
            original_luminance = np.mean(current_split_img)
            remaining_weight = 1.0 - self.smoothing_weight
            for i, prev_frame in enumerate(reversed(self.previous_frames)):
                if i >= self.smoothing_frames - 1:
                    break
                weight = remaining_weight * (0.7 ** i)
                smoothed_img += prev_frame.astype(float) * weight
            smoothed_luminance = np.mean(smoothed_img)
            if smoothed_luminance > 0:
                correction_factor = original_luminance / smoothed_luminance
                smoothed_img *= correction_factor
            split_img = np.clip(smoothed_img, 0, 255).astype(np.uint8)
        else:
            split_img = current_split_img

        self.previous_frames.append(current_split_img)
        if len(self.previous_frames) > self.smoothing_frames:
            self.previous_frames.pop(0)

        return split_img

    def process_video(self, video_path, output_path=None, skip_interval=31):
        """
        Traite une vidéo en effectuant l'inférence sur 1 frame sur 'skip_interval'
        et en générant par interpolation les frames intermédiaires.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Vidéo: {frame_width}x{frame_height}, {fps_video} FPS, {total_frames} frames")

        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps_video, (frame_width, frame_height))

        heavy_times = []  # Durée de traitement heavy (process_frame)
        processed_count = 0
        frame_index = 0
        prev_processed_split = None

        start_time_total = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % skip_interval == 0:
                    # Traitement heavy (inférence)
                    start_heavy = time.time()
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    current_split = self.process_frame(frame_rgb)
                    end_heavy = time.time()
                    heavy_duration = end_heavy - start_heavy
                    heavy_times.append(heavy_duration)

                    # Si une heavy frame précédente existe, générer les frames intermédiaires
                    if prev_processed_split is not None and video_writer:
                        prev_bgr = cv2.cvtColor(prev_processed_split, cv2.COLOR_RGB2BGR)
                        current_bgr = cv2.cvtColor(current_split, cv2.COLOR_RGB2BGR)
                        # Générer par interpolation
                        for j in range(1, skip_interval):
                            alpha = j / float(skip_interval)
                            inter_frame = cv2.addWeighted(prev_bgr, 1 - alpha, current_bgr, alpha, 0)
                            video_writer.write(inter_frame)

                    # Écrire la heavy frame traitée
                    if video_writer:
                        current_bgr = cv2.cvtColor(current_split, cv2.COLOR_RGB2BGR)
                        video_writer.write(current_bgr)

                    prev_processed_split = current_split
                    processed_count += 1

                    if processed_count % 5 == 0:
                        avg_heavy = sum(heavy_times[-5:]) / 5
                        effective_fps = 1.0 / avg_heavy if avg_heavy > 0 else 0
                        print(
                            f"Frame heavy #{processed_count}: durée ~{avg_heavy * 1000:.1f} ms, heavy FPS: {effective_fps:.1f}")

                frame_index += 1

            end_time_total = time.time()
            total_processing_time = end_time_total - start_time_total

        finally:
            cap.release()
            if video_writer:
                video_writer.release()

        # Affichage des statistiques globales
        if heavy_times:
            overall_avg_heavy = sum(heavy_times) / len(heavy_times)
            overall_heavy_fps = 1.0 / overall_avg_heavy if overall_avg_heavy > 0 else 0
            print("\n==== STATISTIQUES DE PERFORMANCE ====")
            print(f"Frames heavy traitées: {processed_count} sur {total_frames} frames")
            print(f"Temps total (heavy + interpolation): {total_processing_time:.2f} s")
            print(f"Durée moyenne heavy: {overall_avg_heavy * 1000:.1f} ms (FPS heavy: {overall_heavy_fps:.1f})")
        if output_path:
            print(f"Vidéo de sortie enregistrée dans: {output_path}")


# Point d'entrée principal
if __name__ == "__main__":
    # Chemins des dossiers
    video_inputs_dir = os.path.join(base_dir, "video_inputs")
    video_outputs_dir = os.path.join(base_dir, "video_outputs")
    os.makedirs(video_outputs_dir, exist_ok=True)

    # Initialiser le synthétiseur
    synthesizer = VideoSplitSynthesizer()

    # Demander la qualité
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

    # Demander le chemin de la vidéo
    video_name = input("Nom de la vidéo à traiter (dans video_inputs/): ")
    video_path = os.path.join(video_inputs_dir, video_name)

    # Demander si on veut sauvegarder la sortie
    save_output = input("Voulez-vous enregistrer la vidéo de sortie? (o/n): ")
    output_path = None
    if save_output.lower() in ["o", "oui"]:
        output_name = f"processed_{video_name}"
        output_path = os.path.join(video_outputs_dir, output_name)

    # Spécifier l'intervalle de frames à traiter
    skip_interval = 31  # Valeur par défaut
    try:
        skip_input = input(f"Intervalle de frames à traiter (défaut: {skip_interval}): ")
        if skip_input.strip():
            skip_interval = int(skip_input)
    except ValueError:
        print(f"Valeur invalide, utilisation de l'intervalle par défaut: {skip_interval}")

    # Paramètres d'interpolation
    interpolation_option = input("Activer l'interpolation entre les couleurs? (o/n, défaut: o): ")
    synthesizer.use_interpolation = interpolation_option.lower() not in ["n", "non"]

    if synthesizer.use_interpolation:
        try:
            interp_strength = input(f"Force d'interpolation (1-20, défaut: {synthesizer.interpolation_strength}): ")
            if interp_strength.strip():
                synthesizer.interpolation_strength = max(1, min(20, int(interp_strength)))
        except ValueError:
            pass

    # Traiter la vidéo
    print(f"Démarrage du traitement de la vidéo avec un intervalle de {skip_interval} frames...")
    synthesizer.process_video(video_path, output_path, skip_interval)