"""
================================================================================
 📄 Script : video_frameskip_interpolator.py
 📦 Projet : Traitement vidéo avec interpolation de frames
 🧠 Objectif :
     Ce script permet de lire une vidéo en entrée et de générer une vidéo en sortie
     en ne traitant qu'une frame sur un intervalle défini (par exemple 1 sur 31),
     et en générant les frames intermédiaires par interpolation pour obtenir
     une animation fluide.

     Ce framework offre une base réutilisable pour différents types de traitements
     vidéo, sans inclure de traitement spécifique.

 🧰 Fonctionnalités :
     - Lecture d'une vidéo depuis un dossier d'entrée
     - Traitement uniquement d'une frame sur un ensemble (exemple : 1 sur 31)
     - Interpolation linéaire pour générer les frames intermédiaires
     - Lissage temporel optionnel
     - Génération d'une vidéo en sortie
     - Affichage des statistiques de performance

 📎 Répertoires utilisés :
     - video_inputs/ : contient les vidéos d'entrée à traiter
     - video_outputs/ : vidéos générées automatiquement

 📎 Dépendances :
     - opencv-python
     - numpy
================================================================================
"""

import cv2
import numpy as np
import time
import os
import sys


class VideoFrameSkipInterpolator:
    def __init__(self):
        """Initialisation du processeur vidéo avec frameskip et interpolation"""
        # Paramètres de configuration
        self.use_interpolation = True
        self.temporal_smoothing = True
        self.smoothing_frames = 5
        self.smoothing_weight = 0.6
        self.previous_frames = []

        # Préréglage de qualité
        self.quality_preset = "balanced"  # options: "performance", "balanced", "quality"
        self.update_quality_settings()

        print(f"Initialisé avec qualité: {self.quality_preset}")

    def update_quality_settings(self):
        """Met à jour les paramètres en fonction du préréglage de qualité choisi"""
        if self.quality_preset == "performance":
            self.processing_resolution = (384, 384)
            self.smoothing_frames = 3
            self.smoothing_weight = 0.5
        elif self.quality_preset == "balanced":
            self.processing_resolution = (512, 512)
            self.smoothing_frames = 5
            self.smoothing_weight = 0.6
        else:  # "quality"
            self.processing_resolution = (768, 768)
            self.smoothing_frames = 8
            self.smoothing_weight = 0.7

        print(f"Paramètres mis à jour: Qualité={self.quality_preset}, Lissage={self.smoothing_frames} frames")

    def preprocess_frame(self, frame):
        """
        Prétraitement de la frame - peut être étendu pour des traitements spécifiques
        Par défaut, ne fait qu'un redimensionnement pour assurer une taille constante
        """
        # Adaptez cette fonction selon vos besoins spécifiques
        return cv2.resize(frame, self.processing_resolution)

    def process_frame(self, frame):
        """
        Traite une frame - applique un flou et augmente la saturation
        """
        # Pré-traitement
        processed = self.preprocess_frame(frame)

        # Conversion en HSV pour manipuler la saturation
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)

        # Augmenter la saturation
        saturation_factor = 1.5  # Valeur > 1 pour augmenter la saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255).astype(np.uint8)

        # Reconversion en BGR
        saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Appliquer le flou
        blur_size = 15  # Taille du noyau pour le flou gaussien (impair)
        blurred = cv2.GaussianBlur(saturated, (blur_size, blur_size), 0)

        # Redimensionnement à la taille d'origine
        processed_resized = cv2.resize(blurred, (frame.shape[1], frame.shape[0]))

        # Lissage temporel si activé
        if self.temporal_smoothing and self.previous_frames:
            smoothed_img = processed_resized.astype(float) * self.smoothing_weight
            remaining_weight = 1.0 - self.smoothing_weight

            # Appliquer un poids décroissant aux frames précédentes
            for i, prev_frame in enumerate(reversed(self.previous_frames)):
                if i >= self.smoothing_frames - 1:
                    break
                weight = remaining_weight * (0.7 ** i)
                smoothed_img += prev_frame.astype(float) * weight

            result = np.clip(smoothed_img, 0, 255).astype(np.uint8)
        else:
            result = processed_resized

        # Mise à jour de la liste des frames précédentes pour le lissage
        self.previous_frames.append(processed_resized)
        if len(self.previous_frames) > self.smoothing_frames:
            self.previous_frames.pop(0)

        return result

    def process_video(self, video_path, output_path=None, skip_interval=31):
        """
        Traite une vidéo en effectuant le traitement sur 1 frame sur 'skip_interval'
        et en générant par interpolation les frames intermédiaires.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
            return

        # Récupérer les propriétés de la vidéo
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Vidéo: {frame_width}x{frame_height}, {fps_video} FPS, {total_frames} frames")

        # Initialiser le writer pour la vidéo de sortie
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps_video, (frame_width, frame_height))

        # Variables de suivi des performances
        processing_times = []  # Durée de traitement des frames clés
        processed_count = 0
        frame_index = 0
        prev_processed_frame = None

        start_time_total = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % skip_interval == 0:
                    # Traitement de la frame clé
                    start_processing = time.time()
                    current_processed = self.process_frame(frame)
                    end_processing = time.time()
                    processing_duration = end_processing - start_processing
                    processing_times.append(processing_duration)

                    # Si une frame clé précédente existe, générer les frames intermédiaires
                    if prev_processed_frame is not None and video_writer:
                        # Générer les frames intermédiaires par interpolation
                        for j in range(1, skip_interval):
                            alpha = j / float(skip_interval)
                            inter_frame = cv2.addWeighted(prev_processed_frame, 1 - alpha, current_processed, alpha, 0)
                            video_writer.write(inter_frame)

                    # Écrire la frame clé traitée
                    if video_writer:
                        video_writer.write(current_processed)

                    prev_processed_frame = current_processed
                    processed_count += 1

                    # Afficher les statistiques toutes les 5 frames clés traitées
                    if processed_count % 5 == 0:
                        avg_processing = sum(processing_times[-5:]) / 5
                        effective_fps = 1.0 / avg_processing if avg_processing > 0 else 0
                        print(
                            f"Frame #{processed_count}/{total_frames//skip_interval}: "
                            f"durée ~{avg_processing * 1000:.1f} ms, FPS: {effective_fps:.1f}"
                        )

                frame_index += 1
                # Afficher la progression
                if frame_index % 100 == 0:
                    progress = (frame_index / total_frames) * 100
                    print(f"Progression: {progress:.1f}%")

            end_time_total = time.time()
            total_processing_time = end_time_total - start_time_total

        finally:
            cap.release()
            if video_writer:
                video_writer.release()

        # Affichage des statistiques globales
        if processing_times:
            overall_avg_processing = sum(processing_times) / len(processing_times)
            overall_processing_fps = 1.0 / overall_avg_processing if overall_avg_processing > 0 else 0

            print("\n==== STATISTIQUES DE PERFORMANCE ====")
            print(f"Frames clés traitées: {processed_count} sur {total_frames} frames")
            print(f"Temps total: {total_processing_time:.2f} s")
            print(f"Durée moyenne par frame clé: {overall_avg_processing * 1000:.1f} ms")
            print(f"FPS des frames clés: {overall_processing_fps:.1f}")
            print(f"FPS effectif (avec interpolation): {(frame_index/total_processing_time):.1f}")

        if output_path:
            print(f"Vidéo de sortie enregistrée dans: {output_path}")


# Point d'entrée principal
if __name__ == "__main__":
    # Récupérer le chemin du projet (racine)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Chemins des dossiers de vidéos (en dehors du dossier courant)
    video_inputs_dir = os.path.join(base_dir, "video_inputs")
    video_outputs_dir = os.path.join(base_dir, "video_outputs")
    os.makedirs(video_outputs_dir, exist_ok=True)

    # Initialiser le processeur
    processor = VideoFrameSkipInterpolator()

    # Demander la qualité
    print("\nChoisissez le préréglage de qualité:")
    print("1 - Performance (plus rapide, qualité moindre)")
    print("2 - Équilibré (recommandé)")
    print("3 - Qualité (plus lent, meilleure qualité)")
    quality_choice = input("Votre choix (1, 2 ou 3): ")
    if quality_choice == "1":
        processor.quality_preset = "performance"
    elif quality_choice == "3":
        processor.quality_preset = "quality"
    else:
        processor.quality_preset = "balanced"
    processor.update_quality_settings()

    # Demander le chemin de la vidéo
    video_name = input("Nom de la vidéo à traiter (dans video_inputs/): ")
    video_path = os.path.join(video_inputs_dir, video_name)

    if not os.path.exists(video_path):
        print(f"Erreur: Le fichier {video_path} n'existe pas.")
        sys.exit(1)

    # Demander si on veut sauvegarder la sortie
    save_output = input("Voulez-vous enregistrer la vidéo de sortie? (o/n): ")
    output_path = None
    if save_output.lower() in ["o", "oui", "y", "yes"]:
        output_name = f"skipped_{video_name}"
        output_path = os.path.join(video_outputs_dir, output_name)

    # Spécifier l'intervalle de frames à traiter
    skip_interval = 31  # Valeur par défaut
    try:
        skip_input = input(f"Intervalle de frames à traiter (défaut: {skip_interval}): ")
        if skip_input.strip():
            skip_interval = int(skip_input)
    except ValueError:
        print(f"Valeur invalide, utilisation de l'intervalle par défaut: {skip_interval}")

    # Paramètres de lissage temporel
    temporal_option = input("Activer le lissage temporel? (o/n, défaut: o): ")
    processor.temporal_smoothing = temporal_option.lower() not in ["n", "non", "no"]

    if processor.temporal_smoothing:
        try:
            smooth_frames = input(f"Nombre de frames pour le lissage (défaut: {processor.smoothing_frames}): ")
            if smooth_frames.strip():
                processor.smoothing_frames = max(2, min(10, int(smooth_frames)))

            smooth_weight = input(f"Poids du lissage (0.1-0.9, défaut: {processor.smoothing_weight}): ")
            if smooth_weight.strip():
                processor.smoothing_weight = max(0.1, min(0.9, float(smooth_weight)))
        except ValueError:
            pass

    # Traiter la vidéo
    print(f"\nDémarrage du traitement de la vidéo avec un intervalle de {skip_interval} frames...")
    processor.process_video(video_path, output_path, skip_interval)