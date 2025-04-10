"""
================================================================================
 📄 Module : visualization_utils.py
 📦 Projet : New Chameleon Strategy – Phase finale image
 🧠 Objectif :
     Fournir les fonctions de visualisation graphique pour les résultats
     d’analyse de couleur et de segmentation.

 🧰 Fonctionnalités :
     - Affichage de l’image originale et du masque segmenté
     - Barres horizontales pour les couleurs dominantes par classe
     - Légende dynamique des classes présentes
     - Support des proportions et de l’interpolation visuelle

 📎 Utilisé par :
     - analyze_image.py

 📎 Dépendances :
     - matplotlib
     - numpy
     - opencv-python

 🧪 Auteur : Hashif Batcha
================================================================================
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
from sklearn.cluster import KMeans
import matplotlib
import os

matplotlib.rcParams.update({'font.size': 9})  # Réduire légèrement la taille de police par défaut


def visualize_segmentation_results(image, mask, cityscapes_colors, cityscapes_classes, classes_of_interest,
                                   colors_per_class):
    """
    Fonction pour créer une visualisation des résultats de segmentation avec un nombre de couleurs personnalisé par classe

    Args:
        image: Image originale (RGB)
        mask: Masque de segmentation
        cityscapes_colors: Liste des couleurs pour chaque classe
        cityscapes_classes: Dictionnaire des noms de classes
        classes_of_interest: Dictionnaire des classes d'intérêt à analyser
        colors_per_class: Dictionnaire spécifiant le nombre de couleurs à extraire pour chaque classe
                         (globale et pour les parties gauche/droite)
    """

    # Obtenir les classes présentes
    classes_present = np.unique(mask)

    # Créer la segmentation colorée
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(cityscapes_colors):
        colored_mask[mask == class_id] = color

    # Créer une figure avec un layout fixe
    fig = plt.figure(figsize=(21, 14))

    # Utiliser une approche de positionnement absolu pour un meilleur contrôle
    # 1. Image originale
    ax_img = fig.add_axes([0.05, 0.6, 0.4, 0.3])
    ax_img.imshow(image)
    ax_img.set_title("Image originale", fontsize=14, fontweight='bold')
    ax_img.axis('off')

    # 2. Segmentation
    ax_seg = fig.add_axes([0.55, 0.6, 0.4, 0.3])
    ax_seg.imshow(colored_mask)
    ax_seg.set_title("Segmentation sémantique", fontsize=14, fontweight='bold')
    ax_seg.axis('off')

    # 3. Statistiques générales
    ax_stats = fig.add_axes([0.05, 0.45, 0.4, 0.13])
    ax_stats.set_title("Statistiques des classes", fontsize=14, fontweight='bold')

    # Calculer les pourcentages de chaque classe
    total_pixels = mask.size
    classes_stats = []
    for class_id in classes_present:
        n_pixels = np.sum(mask == class_id)
        percentage = (n_pixels / total_pixels) * 100
        classes_stats.append((class_id, percentage))

    # Trier par pourcentage décroissant
    classes_stats.sort(key=lambda x: x[1], reverse=True)

    # Afficher les statistiques sous forme de barres horizontales
    y_pos = np.arange(len(classes_stats))
    percentages = [pct for _, pct in classes_stats]
    class_colors = [np.array(cityscapes_colors[cid]) / 255 for cid, _ in classes_stats]

    # Créer les barres avec les noms des classes intégrés dans les barres
    bars = ax_stats.barh(y_pos, percentages, color=class_colors)

    # Créer les étiquettes avec formatage en gras pour les classes d'intérêt
    class_labels = []
    for cid, _ in classes_stats:
        class_name = cityscapes_classes[cid]
        if cid in classes_of_interest:
            # Utiliser la syntaxe de Matplotlib pour le gras avec $\mathbf{texte}$
            # ou alternativement, créer des objets texte différents plus loin
            class_labels.append(f"{class_name} (Classe {cid})")
        else:
            class_labels.append(f"{class_name} (Classe {cid})")

    # Ajouter les noms des classes sur l'axe y (à gauche des barres)
    ax_stats.set_yticks(y_pos)
    ax_stats.set_yticklabels(class_labels)

    # Appliquer le style gras manuellement après avoir défini les étiquettes
    for i, (cid, _) in enumerate(classes_stats):
        if cid in classes_of_interest:
            ax_stats.get_yticklabels()[i].set_fontweight('bold')

    # Ajouter les valeurs du pourcentage à l'extrémité des barres
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        ax_stats.text(pct + 0.5, bar.get_y() + bar.get_height() / 2,
                      f'{pct:.1f}%', va='center', fontsize=8)

    # Configuration de l'axe x pour afficher les pourcentages
    ax_stats.set_xlabel('Pourcentage de l\'image (%)')
    ax_stats.set_xlim(0, max(percentages) * 1.1)  # Échelle adaptative comme à l'origine

    # 4. Légende complète
    ax_legend = fig.add_axes([0.55, 0.45, 0.4, 0.13])
    ax_legend.axis('off')
    ax_legend.set_title("Classes d'intérêt analysées", fontsize=14, fontweight='bold')

    # Séparation de la légende en colonnes pour toutes les classes présentes
    n_classes = len(classes_present)
    n_cols = 3  # Nombre de colonnes pour la légende
    n_rows = (n_classes + n_cols - 1) // n_cols  # Calculer nombre de lignes nécessaires

    # Positionnement pour la légende en grille
    x_positions = [0.05, 0.35, 0.65]  # Positions X pour les 3 colonnes

    for i, class_id in enumerate(classes_present):
        # Calculer position (ligne, colonne)
        col = i % n_cols
        row = i // n_cols

        # Calculer coordonnées
        x = x_positions[col]
        y = 0.9 - (row * 0.2)

        # Afficher un rectangle coloré (plus grand)
        color = np.array(cityscapes_colors[class_id]) / 255
        rect = Rectangle((x, y - 0.05), 0.08, 0.08,  # Rectangle plus grand
                         facecolor=color, edgecolor='black')
        ax_legend.add_patch(rect)

        # Préparer le texte (en gras si classe d'intérêt)
        class_name = cityscapes_classes[class_id]
        is_interest = class_id in classes_of_interest

        # Afficher le texte (avec formatage gras si nécessaire)
        if is_interest:
            ax_legend.text(x + 0.1, y - 0.01, f"{class_name} (Classe {class_id})",
                           va='center', fontsize=10, fontweight='bold')
        else:
            ax_legend.text(x + 0.1, y - 0.01, f"{class_name} (Classe {class_id})",
                           va='center', fontsize=10)

    # 5. Titre central pour l'analyse des couleurs
    ax_title = fig.add_axes([0.0, 0.35, 1.0, 0.05])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, "Analyse des couleurs dominantes par classe",
                  fontsize=16, fontweight='bold', ha='center', va='center')

    # 6. Afficher les analyses de couleurs pour chaque classe d'intérêt
    interest_present = [cid for cid in classes_of_interest.keys() if cid in classes_present]
    n_interest = len(interest_present)

    if n_interest > 0:
        # Positionner les analyses de couleur
        if n_interest == 1:
            class_id = interest_present[0]
            class_name = classes_of_interest[class_id]
            ax_class = fig.add_axes([0.3, 0.05, 0.4, 0.3])
            display_class_colors(ax_class, image, mask, class_id, class_name, colors_per_class)
        elif n_interest == 2:
            positions = [[0.05, 0.05, 0.4, 0.3], [0.55, 0.05, 0.4, 0.3]]
            for i, class_id in enumerate(interest_present):
                class_name = classes_of_interest[class_id]
                ax_class = fig.add_axes(positions[i])
                display_class_colors(ax_class, image, mask, class_id, class_name, colors_per_class)
        else:
            positions = [[0.05, 0.05, 0.3, 0.3], [0.375, 0.05, 0.3, 0.3], [0.7, 0.05, 0.3, 0.3]]
            for i, class_id in enumerate(interest_present[:3]):  # Limiter à 3 classes
                class_name = classes_of_interest[class_id]
                ax_class = fig.add_axes(positions[i])
                display_class_colors(ax_class, image, mask, class_id, class_name, colors_per_class)

    return fig


def display_class_colors(ax, image, mask, class_id, class_name, colors_per_class):
    """Affiche les couleurs dominantes pour une classe spécifique"""
    ax.axis('off')
    ax.set_title(class_name, fontsize=14, fontweight='bold')

    if colors_per_class and class_id in colors_per_class:
        n_colors_global = colors_per_class[class_id].get('global')
        n_colors_split = colors_per_class[class_id].get('split')

    # Analyse des couleurs dominantes
    colors_full, props_full = extract_dominant_colors(image, mask, class_id, n_colors=n_colors_global)
    left_colors, left_props, right_colors, right_props = extract_dominant_colors_split(
        image, mask, class_id, n_colors=n_colors_split)

    # Y positions pour l'affichage
    y_pos = 0.95
    rect_height = 0.04  # Hauteur du rectangle réduite
    rect_width = 0.06
    y_offset = 0.05  # Espacement entre éléments
    section_offset = 0.09  # Espacement supplémentaire entre les sections (augmenté)

    # Calcul du nombre total maximum de couleurs à afficher
    max_colors_global = 0 if colors_full is None else len(colors_full)
    max_colors_left = 0 if left_colors is None else len(left_colors)
    max_colors_right = 0 if right_colors is None else len(right_colors)

    # Adapter l'espacement en fonction du nombre total d'éléments à afficher
    total_items = max_colors_global + max_colors_left + max_colors_right + 6
    if total_items > 16:
        y_offset = 0.035  # Réduire l'espacement si beaucoup d'éléments
        section_offset = 0.07  # Réduire aussi l'espacement entre sections mais garder plus grand que y_offset

    # 1. Couleurs globales
    ax.text(0.05, y_pos, "Couleurs dominantes globales:", fontsize=10, fontweight='bold')
    y_pos -= y_offset

    if colors_full is not None:
        for i, (color, prop) in enumerate(zip(colors_full, props_full)):
            # Rectangle coloré
            rect = Rectangle((0.05, y_pos - rect_height), rect_width, rect_height,
                             facecolor=color / 255, edgecolor='black')
            ax.add_patch(rect)

            # Format RGB propre, convertir explicitement les valeurs NumPy en int Python standard
            rgb_str = f"RGB[{int(color[0])}, {int(color[1])}, {int(color[2])}]"
            ax.text(0.05 + rect_width + 0.02, y_pos - rect_height / 2,
                    f"{rgb_str} - {prop * 100:.1f}%",
                    va='center', fontsize=9)
            y_pos -= y_offset
    else:
        ax.text(0.05 + rect_width + 0.02, y_pos - rect_height / 2, "Pas de couleurs dominantes.", fontsize=9)
        y_pos -= y_offset

    # Augmenter l'espace entre les sections
    y_pos -= section_offset

    # 2. Partie gauche
    ax.text(0.05, y_pos, "Partie gauche:", fontsize=10, fontweight='bold')
    y_pos -= y_offset

    if left_colors is not None:
        for i, (color, prop) in enumerate(zip(left_colors, left_props)):
            # Rectangle coloré
            rect = Rectangle((0.05, y_pos - rect_height), rect_width, rect_height,
                             facecolor=color / 255, edgecolor='black')
            ax.add_patch(rect)

            # Format RGB propre
            rgb_str = f"RGB[{int(color[0])}, {int(color[1])}, {int(color[2])}]"
            ax.text(0.05 + rect_width + 0.02, y_pos - rect_height / 2,
                    f"{rgb_str} - {prop * 100:.1f}%",
                    va='center', fontsize=9)
            y_pos -= y_offset
    else:
        ax.text(0.05 + rect_width + 0.02, y_pos - rect_height / 2, "Pas de couleurs dominantes.", fontsize=9)
        y_pos -= y_offset

    # Augmenter l'espace entre les sections
    y_pos -= section_offset

    # 3. Partie droite
    ax.text(0.05, y_pos, "Partie droite:", fontsize=10, fontweight='bold')
    y_pos -= y_offset

    if right_colors is not None:
        for i, (color, prop) in enumerate(zip(right_colors, right_props)):
            # Rectangle coloré
            rect = Rectangle((0.05, y_pos - rect_height), rect_width, rect_height,
                             facecolor=color / 255, edgecolor='black')
            ax.add_patch(rect)

            # Format RGB propre
            rgb_str = f"RGB[{int(color[0])}, {int(color[1])}, {int(color[2])}]"
            ax.text(0.05 + rect_width + 0.02, y_pos - rect_height / 2,
                    f"{rgb_str} - {prop * 100:.1f}%",
                    va='center', fontsize=9)
            y_pos -= y_offset
    else:
        ax.text(0.05 + rect_width + 0.02, y_pos - rect_height / 2, "Pas de couleurs dominantes.", fontsize=9)
        y_pos -= y_offset


def extract_dominant_colors(image, mask, class_id, n_colors=5):
    """Extraire les couleurs dominantes pour une classe"""
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

    # Éviter une erreur si le nombre de pixels est inférieur à n_colors
    if len(pixels) < n_colors:
        n_colors = max(1, len(pixels))

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
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


def extract_dominant_colors_split(image, mask, class_id, n_colors=3):
    """Extraire les couleurs dominantes pour chaque moitié de l'image"""
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

    left_colors = left_proportions = right_colors = right_proportions = None

    # Traiter la partie gauche si elle contient des pixels
    if np.any(left_mask):
        left_pixels = image_resized[left_mask]

        # Éviter une erreur si le nombre de pixels est inférieur à n_colors
        n_left_colors = min(n_colors, len(left_pixels))
        if n_left_colors > 0:
            kmeans_left = KMeans(n_clusters=n_left_colors, random_state=42)
            kmeans_left.fit(left_pixels)
            left_colors = kmeans_left.cluster_centers_.astype(int)
            left_labels = kmeans_left.labels_
            left_proportions = np.bincount(left_labels) / len(left_labels)

            # Trier par proportion
            idx = np.argsort(-left_proportions)
            left_colors = left_colors[idx]
            left_proportions = left_proportions[idx]

    # Traiter la partie droite si elle contient des pixels
    if np.any(right_mask):
        right_pixels = image_resized[right_mask]

        # Éviter une erreur si le nombre de pixels est inférieur à n_colors
        n_right_colors = min(n_colors, len(right_pixels))
        if n_right_colors > 0:
            kmeans_right = KMeans(n_clusters=n_right_colors, random_state=42)
            kmeans_right.fit(right_pixels)
            right_colors = kmeans_right.cluster_centers_.astype(int)
            right_labels = kmeans_right.labels_
            right_proportions = np.bincount(right_labels) / len(right_labels)

            # Trier par proportion
            idx = np.argsort(-right_proportions)
            right_colors = right_colors[idx]
            right_proportions = right_proportions[idx]

    return left_colors, left_proportions, right_colors, right_proportions


# Fonction pour créer un tableau de résultats pour l'export
# Modification de la fonction create_results_table pour inclure les détails des proportions

def create_results_table(image, mask, classes_of_interest, classes_present, cityscapes_classes, colors_per_class):
    """Générer un tableau de résultats exportable avec format RGB amélioré et détails des proportions"""
    results = []

    # Définir les valeurs par défaut pour le nombre de couleurs
    if colors_per_class is None:
        colors_per_class = {
            8: {'global': 5, 'split': 3},  # végétation: 5 couleurs globales, 3 par côté
            9: {'global': 3, 'split': 3},  # terrain: 3 couleurs globales, 3 par côté
            10: {'global': 2, 'split': 2},  # ciel: 2 couleurs globales, 2 par côté
        }

    # Ajouter en-tête
    results.append("ANALYSE DES COULEURS DOMINANTES PAR CLASSE\n")
    results.append("=" * 60 + "\n\n")

    # Statistiques globales
    total_pixels = mask.size
    results.append("STATISTIQUES GLOBALES:\n")
    results.append("-" * 30 + "\n")

    # Dictionnaire pour stocker les statistiques des classes
    class_stats = {}

    for class_id in classes_present:
        n_pixels = np.sum(mask == class_id)
        percentage = (n_pixels / total_pixels) * 100
        class_name = cityscapes_classes[class_id]
        results.append(f"{class_name} (Classe {class_id}): {percentage:.2f}%\n")

        # Stocker les statistiques pour les classes d'intérêt
        if class_id in classes_of_interest:
            class_stats[class_id] = {
                'name': class_name,
                'pixels': n_pixels,
                'percentage': percentage
            }

    # Calculer les proportions relatives entre les classes d'intérêt seulement
    total_interest_pixels = sum(stats['pixels'] for stats in class_stats.values())
    total_interest_percentage = sum(stats['percentage'] for stats in class_stats.values())

    results.append("\nPROPORTIONS RELATIVES ENTRE CLASSES D'INTÉRÊT:\n")
    results.append("-" * 50 + "\n")
    results.append(
        f"Total des pixels des classes d'intérêt: {total_interest_pixels} pixels ({total_interest_percentage:.2f}% de l'image)\n\n")

    for class_id, stats in class_stats.items():
        relative_percentage = (stats['pixels'] / total_interest_pixels) * 100
        results.append(f"{stats['name']} (Classe {class_id}):\n")
        results.append(f"  - Pixels absolus: {stats['pixels']} pixels\n")
        results.append(f"  - Pourcentage absolu: {stats['percentage']:.2f}% de l'image totale\n")
        results.append(f"  - Pourcentage relatif: {relative_percentage:.2f}% des classes d'intérêt\n")
        results.append(f"  - Hauteur relative dans l'image de synthèse: {relative_percentage:.2f}%\n\n")

    # Analyse détaillée pour les classes d'intérêt
    results.append("\n")

    # Diviser l'image en moitiés gauche et droite pour le calcul des pixels
    h, w = mask.shape
    mid_w = w // 2
    left_mask = mask[:, :mid_w]
    right_mask = mask[:, mid_w:]

    # NOUVEAU: Calculer les totaux gauche/droite pour les classes d'intérêt
    left_interest_pixels = 0
    right_interest_pixels = 0
    for class_id in classes_of_interest:
        if class_id in classes_present:
            left_interest_pixels += np.sum(left_mask == class_id)
            right_interest_pixels += np.sum(right_mask == class_id)

    for class_id, class_name in classes_of_interest.items():
        if class_id in classes_present:
            results.append(f"ANALYSE DÉTAILLÉE: {class_name.upper()} (Classe {class_id})\n")
            results.append("-" * 50 + "\n")

            # Statistiques de répartition gauche/droite
            total_class_pixels = np.sum(mask == class_id)
            left_class_pixels = np.sum(left_mask == class_id)
            right_class_pixels = np.sum(right_mask == class_id)

            left_percentage = (left_class_pixels / total_class_pixels) * 100 if total_class_pixels > 0 else 0
            right_percentage = (right_class_pixels / total_class_pixels) * 100 if total_class_pixels > 0 else 0

            # NOUVEAU: Calculer les pourcentages relatifs pour chaque côté
            left_relative_percentage = (
                                                   left_class_pixels / left_interest_pixels) * 100 if left_interest_pixels > 0 else 0
            right_relative_percentage = (
                                                    right_class_pixels / right_interest_pixels) * 100 if right_interest_pixels > 0 else 0

            results.append(f"Répartition spatiale:\n")
            results.append(f"  - Total: {total_class_pixels} pixels\n")
            results.append(f"  - Partie gauche: {left_class_pixels} pixels ({left_percentage:.2f}% de la classe)\n")
            results.append(f"  - Partie droite: {right_class_pixels} pixels ({right_percentage:.2f}% de la classe)\n\n")

            # NOUVEAU: Ajouter les pourcentages relatifs pour chaque côté
            results.append(f"Pourcentage relatif par côté (pour l'image de synthèse split):\n")
            results.append(f"  - Partie gauche: {left_relative_percentage:.2f}% des classes d'intérêt à gauche\n")
            results.append(f"  - Partie droite: {right_relative_percentage:.2f}% des classes d'intérêt à droite\n\n")

            if class_id in colors_per_class:
                n_colors_global = colors_per_class[class_id].get('global')
                n_colors_split = colors_per_class[class_id].get('split')

            # Couleurs globales
            colors, proportions = extract_dominant_colors(image, mask, class_id, n_colors=n_colors_global)
            if colors is not None:
                results.append("Couleurs dominantes globales:\n")
                for i, (color, prop) in enumerate(zip(colors, proportions)):
                    results.append(
                        f"- Couleur {i + 1}: RGB[{int(color[0])}, {int(color[1])}, {int(color[2])}] - {prop * 100:.1f}%\n")

                results.append("\nHauteurs relatives des bandes de couleur dans l'image de synthèse globale:\n")
                running_total = 0
                for i, prop in enumerate(proportions):
                    start_percent = running_total
                    end_percent = running_total + prop * 100
                    results.append(
                        f"- Couleur {i + 1}: de {start_percent:.1f}% à {end_percent:.1f}% de la hauteur de la bande\n")
                    running_total = end_percent

            # Couleurs gauche/droite
            left_colors, left_props, right_colors, right_props = extract_dominant_colors_split(
                image, mask, class_id, n_colors=n_colors_split)

            results.append("\nPartie gauche:\n")
            if left_colors is not None:
                for i, (color, prop) in enumerate(zip(left_colors, left_props)):
                    results.append(
                        f"- Couleur {i + 1}: RGB[{int(color[0])}, {int(color[1])}, {int(color[2])}] - {prop * 100:.1f}%\n")

                results.append("\nHauteurs relatives des bandes de couleur dans la partie gauche de l'image split:\n")
                running_total = 0
                for i, prop in enumerate(left_props):
                    start_percent = running_total
                    end_percent = running_total + prop * 100
                    results.append(
                        f"- Couleur {i + 1}: de {start_percent:.1f}% à {end_percent:.1f}% de la hauteur de la bande\n")
                    running_total = end_percent
            else:
                results.append("- Aucun pixel de cette classe dans cette partie\n")

            results.append("\nPartie droite:\n")
            if right_colors is not None:
                for i, (color, prop) in enumerate(zip(right_colors, right_props)):
                    results.append(
                        f"- Couleur {i + 1}: RGB[{int(color[0])}, {int(color[1])}, {int(color[2])}] - {prop * 100:.1f}%\n")

                results.append("\nHauteurs relatives des bandes de couleur dans la partie droite de l'image split:\n")
                running_total = 0
                for i, prop in enumerate(right_props):
                    start_percent = running_total
                    end_percent = running_total + prop * 100
                    results.append(
                        f"- Couleur {i + 1}: de {start_percent:.1f}% à {end_percent:.1f}% de la hauteur de la bande\n")
                    running_total = end_percent
            else:
                results.append("- Aucun pixel de cette classe dans cette partie\n")

            results.append("\n" + "=" * 60 + "\n\n")

    return "".join(results)


def create_dominant_color_images(image, mask, cityscapes_classes, colors_per_class=None, output_dir=None,
                                 image_name=None, debug=False):
    """
    Crée deux images de synthèse basées sur les couleurs dominantes des classes d'intérêt (ciel, végétation, terrain)
    Version corrigée pour gérer correctement les différences de dimensions entre l'image et le masque.

    Args:
        image: Image originale
        mask: Masque de segmentation
        cityscapes_classes: Dictionnaire des classes
        colors_per_class: Dictionnaire spécifiant le nombre de couleurs à extraire pour chaque classe
        output_dir: Répertoire de sortie (optionnel)
        image_name: Nom de base pour les fichiers de sortie (optionnel)
        debug: Activer le mode debug pour afficher des messages détaillés

    Returns:
        global_img: Image de synthèse avec les couleurs dominantes globales
        split_img: Image de synthèse avec les couleurs dominantes gauche/droite
    """
    import os
    import numpy as np
    import cv2

    if debug:
        print("Démarrage de la création des images de synthèse")

    # Classes d'intérêt
    classes_of_interest = {
        8: 'végétation',  # vegetation
        9: 'terrain',  # terrain
        10: 'ciel'  # ciel
    }

    # Valeurs par défaut pour le nombre de couleurs si non spécifié
    if colors_per_class is None:
        colors_per_class = {
            8: {'global': 5, 'split': 3},  # végétation: 5 couleurs globales, 3 par côté
            9: {'global': 3, 'split': 3},  # terrain: 3 couleurs globales, 3 par côté
            10: {'global': 2, 'split': 2},  # ciel: 2 couleurs globales, 2 par côté
        }

    # Obtenir les dimensions de l'image et du masque
    h_img, w_img = image.shape[:2]
    h_mask, w_mask = mask.shape

    if debug:
        print(f"Dimensions de l'image originale: {w_img}x{h_img}")
        print(f"Dimensions du masque: {w_mask}x{h_mask}")

    # CORRECTION: Calculer les points milieux séparément pour l'image et le masque
    mid_w_img = w_img // 2  # Pour l'image de synthèse
    mid_w_mask = w_mask // 2  # Pour le masque

    if debug:
        print(f"Point milieu image: {mid_w_img}")
        print(f"Point milieu masque: {mid_w_mask}")

    # Trier les classes d'intérêt par ordre vertical (ciel en haut, puis végétation, puis terrain)
    vertical_order = [10, 8, 9]  # [ciel, végétation, terrain]

    # CORRECTION: Créer des masques booléens basés sur le masque de segmentation
    mask_left = np.zeros_like(mask, dtype=bool)
    mask_right = np.zeros_like(mask, dtype=bool)

    # Définir les régions gauche et droite en utilisant le point milieu du masque
    mask_left[:, :mid_w_mask] = True
    mask_right[:, mid_w_mask:] = True

    if debug:
        print(f"Forme du masque gauche: {mask_left.shape}")
        print(f"Forme du masque droit: {mask_right.shape}")

    # Identifier les classes présentes dans chaque partie
    classes_present_global = []
    classes_present_left = []
    classes_present_right = []

    # Dictionnaires pour stocker les pixels par classe pour chaque partie
    pixels_global = {}
    pixels_left = {}
    pixels_right = {}

    # Extraction des couleurs dominantes pour chaque classe
    global_colors = {}
    left_colors = {}
    right_colors = {}
    global_props = {}
    left_props = {}
    right_props = {}

    # Identifier les classes présentes et extraire les couleurs
    for class_id in classes_of_interest:
        class_mask = mask == class_id

        # Vérifier si la classe est présente globalement
        if np.any(class_mask):
            classes_present_global.append(class_id)
            pixels_global[class_id] = np.sum(class_mask)

            if debug:
                print(f"Classe {class_id} présente globalement: {pixels_global[class_id]} pixels")

            # Obtenir le nombre de couleurs pour cette classe
            n_colors_global = colors_per_class[class_id]['global'] if class_id in colors_per_class else 5

            # Extraire les couleurs dominantes globales
            colors, props = extract_dominant_colors(image, mask, class_id, n_colors=n_colors_global)
            if colors is not None:
                global_colors[class_id] = colors
                global_props[class_id] = props
                if debug:
                    print(f"Couleurs globales extraites pour classe {class_id}")

        # Vérifier si la classe est présente à gauche (en utilisant le masque booléen)
        class_mask_left = np.logical_and(class_mask, mask_left)
        if np.any(class_mask_left):
            classes_present_left.append(class_id)
            pixels_left[class_id] = np.sum(class_mask_left)
            if debug:
                print(f"Classe {class_id} présente à gauche: {pixels_left[class_id]} pixels")

        # Vérifier si la classe est présente à droite (en utilisant le masque booléen)
        class_mask_right = np.logical_and(class_mask, mask_right)
        if np.any(class_mask_right):
            classes_present_right.append(class_id)
            pixels_right[class_id] = np.sum(class_mask_right)
            if debug:
                print(f"Classe {class_id} présente à droite: {pixels_right[class_id]} pixels")

        # Obtenir le nombre de couleurs pour split
        n_colors_split = colors_per_class[class_id]['split'] if class_id in colors_per_class else 3

        # Extraire directement les couleurs pour chaque côté
        # en passant les masques booléens combinés
        if np.any(class_mask_left):
            left_colors[class_id], left_props[class_id] = extract_dominant_colors_direct(
                image, class_mask_left, n_colors=n_colors_split)
            if debug and left_colors[class_id] is not None:
                print(f"Couleurs gauche extraites pour classe {class_id}: {len(left_colors[class_id])} couleurs")

        if np.any(class_mask_right):
            right_colors[class_id], right_props[class_id] = extract_dominant_colors_direct(
                image, class_mask_right, n_colors=n_colors_split)
            if debug and right_colors[class_id] is not None:
                print(f"Couleurs droite extraites pour classe {class_id}: {len(right_colors[class_id])} couleurs")

    # Si aucune classe d'intérêt n'est présente, retourner des images noires
    if len(classes_present_global) == 0:
        if debug:
            print("Aucune classe d'intérêt présente dans l'image")
        global_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        split_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)

        if output_dir and image_name:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(f"{output_dir}/{image_name}_couleurs_globales.jpg", cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{output_dir}/{image_name}_couleurs_split.jpg", cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR))

        return global_img, split_img

    # Trier les classes présentes selon l'ordre vertical
    classes_present_global.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))
    classes_present_left.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))
    classes_present_right.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))

    if debug:
        print(f"Classes triées globalement: {classes_present_global}")
        print(f"Classes triées à gauche: {classes_present_left}")
        print(f"Classes triées à droite: {classes_present_right}")

    # Calculer les proportions relatives pour chaque partie
    # 1. Global
    total_pixels_global = sum(pixels_global.values())
    relative_heights_global = {}
    for class_id in classes_present_global:
        relative_heights_global[class_id] = (pixels_global[class_id] / total_pixels_global) * 100
        if debug:
            print(f"Hauteur relative globale classe {class_id}: {relative_heights_global[class_id]:.2f}%")

    # 2. Gauche
    total_pixels_left = sum(pixels_left.values()) if pixels_left else 0
    relative_heights_left = {}
    if total_pixels_left > 0:
        for class_id in classes_present_left:
            relative_heights_left[class_id] = (pixels_left[class_id] / total_pixels_left) * 100
            if debug:
                print(f"Hauteur relative gauche classe {class_id}: {relative_heights_left[class_id]:.2f}%")

    # 3. Droite
    total_pixels_right = sum(pixels_right.values()) if pixels_right else 0
    relative_heights_right = {}
    if total_pixels_right > 0:
        for class_id in classes_present_right:
            relative_heights_right[class_id] = (pixels_right[class_id] / total_pixels_right) * 100
            if debug:
                print(f"Hauteur relative droite classe {class_id}: {relative_heights_right[class_id]:.2f}%")

    # CORRECTION: Créer les images de synthèse avec les dimensions de l'image originale
    global_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    split_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    # 1. Générer l'image globale
    section_heights_global = []
    remaining_height = h_img  # Utiliser la hauteur de l'image originale

    for i, class_id in enumerate(classes_present_global):
        if i == len(classes_present_global) - 1:
            # La dernière classe prend tout l'espace restant
            section_heights_global.append(remaining_height)
        else:
            # Calculer la hauteur de cette section
            section_height = int((relative_heights_global[class_id] / 100) * h_img)
            section_height = max(1, section_height)  # Minimum 1 pixel
            section_heights_global.append(section_height)
            remaining_height -= section_height

        if debug:
            print(f"Hauteur section globale classe {class_id}: {section_heights_global[-1]} pixels")

    # Dessiner les sections pour l'image globale
    y_pos = 0
    for i, class_id in enumerate(classes_present_global):
        section_height = section_heights_global[i]

        if class_id in global_colors and global_colors[class_id] is not None:
            colors = global_colors[class_id]
            props = global_props[class_id]

            # Normaliser les proportions
            total_prop = sum(props)
            if total_prop > 0:
                normalized_props = props / total_prop

                # Calculer les hauteurs de chaque bande de couleur
                color_heights = []
                remaining = section_height

                for j, prop in enumerate(normalized_props):
                    if j == len(normalized_props) - 1:
                        # La dernière couleur prend l'espace restant
                        color_heights.append(remaining)
                    else:
                        # Calculer la hauteur de cette bande
                        color_height = int(prop * section_height)
                        color_height = max(1, color_height)  # Minimum 1 pixel
                        color_heights.append(color_height)
                        remaining -= color_height

                # Dessiner chaque bande de couleur
                section_y = y_pos
                for j, (color, height) in enumerate(zip(colors, color_heights)):
                    # S'assurer que les dimensions sont valides
                    if height > 0 and section_y < h_img and section_y + height <= h_img:
                        try:
                            # Convertir explicitement la couleur en uint8
                            color_uint8 = np.clip(color, 0, 255).astype(np.uint8)
                            global_img[section_y:section_y + height, :] = color_uint8
                            if debug:
                                print(f"Dessiné bande globale: y={section_y}, h={height}, color={color_uint8}")
                        except Exception as e:
                            if debug:
                                print(f"Erreur lors du dessin global: {e}")
                    else:
                        if debug:
                            print(f"Dimensions invalides pour dessin global: y={section_y}, h={height}")
                    section_y += height

        # Mettre à jour la position verticale
        y_pos += section_height

    # 2. PARTIE GAUCHE - gérer le cas où il n'y a pas de classes à gauche
    if classes_present_left:
        section_heights_left = []
        remaining_height = h_img  # Utiliser la hauteur de l'image originale

        for i, class_id in enumerate(classes_present_left):
            if i == len(classes_present_left) - 1:
                section_heights_left.append(remaining_height)
            else:
                section_height = int((relative_heights_left[class_id] / 100) * h_img)
                section_height = max(1, section_height)
                section_heights_left.append(section_height)
                remaining_height -= section_height

            if debug:
                print(f"Hauteur section gauche classe {class_id}: {section_heights_left[-1]} pixels")

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

                    # Calculer les hauteurs de chaque bande de couleur
                    color_heights = []
                    remaining = section_height

                    for j, prop in enumerate(normalized_props):
                        if j == len(normalized_props) - 1:
                            color_heights.append(remaining)
                        else:
                            color_height = int(prop * section_height)
                            color_height = max(1, color_height)
                            color_heights.append(color_height)
                            remaining -= color_height

                    # Dessiner chaque bande de couleur
                    section_y = y_pos_left
                    for j, (color, height) in enumerate(zip(colors, color_heights)):
                        # S'assurer que les dimensions sont valides
                        if height > 0 and section_y < h_img and section_y + height <= h_img:
                            try:
                                # Convertir explicitement la couleur en uint8
                                color_uint8 = np.clip(color, 0, 255).astype(np.uint8)
                                # CORRECTION: Utiliser mid_w_img pour dessiner dans la moitié gauche
                                split_img[section_y:section_y + height, :mid_w_img] = color_uint8
                                if debug:
                                    print(f"Dessiné bande gauche: y={section_y}, h={height}, color={color_uint8}")
                            except Exception as e:
                                if debug:
                                    print(f"Erreur lors du dessin gauche: {e}")
                        else:
                            if debug:
                                print(f"Dimensions invalides pour dessin gauche: y={section_y}, h={height}")
                        section_y += height

            # Mettre à jour la position verticale
            y_pos_left += section_height

    # 3. PARTIE DROITE - gérer le cas où il n'y a pas de classes à droite
    if classes_present_right:
        section_heights_right = []
        remaining_height = h_img  # Utiliser la hauteur de l'image originale

        for i, class_id in enumerate(classes_present_right):
            if i == len(classes_present_right) - 1:
                section_heights_right.append(remaining_height)
            else:
                # Utiliser les proportions relatives de la partie droite
                section_height = int((relative_heights_right[class_id] / 100) * h_img)
                section_height = max(1, section_height)
                section_heights_right.append(section_height)
                remaining_height -= section_height

            if debug:
                print(f"Hauteur section droite classe {class_id}: {section_heights_right[-1]} pixels")

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

                    # Calculer les hauteurs de chaque bande de couleur
                    color_heights = []
                    remaining = section_height

                    for j, prop in enumerate(normalized_props):
                        if j == len(normalized_props) - 1:
                            color_heights.append(remaining)
                        else:
                            color_height = int(prop * section_height)
                            color_height = max(1, color_height)
                            color_heights.append(color_height)
                            remaining -= color_height

                    # Dessiner chaque bande de couleur
                    section_y = y_pos_right
                    for j, (color, height) in enumerate(zip(colors, color_heights)):
                        # S'assurer que les dimensions sont valides
                        if height > 0 and section_y < h_img and section_y + height <= h_img:
                            try:
                                # Convertir explicitement la couleur en uint8
                                color_uint8 = np.clip(color, 0, 255).astype(np.uint8)
                                # CORRECTION: Utiliser mid_w_img pour dessiner dans la moitié droite
                                split_img[section_y:section_y + height, mid_w_img:] = color_uint8
                                if debug:
                                    print(f"Dessiné bande droite: y={section_y}, h={height}, color={color_uint8}")
                            except Exception as e:
                                if debug:
                                    print(f"Erreur lors du dessin droite: {e}")
                        else:
                            if debug:
                                print(f"Dimensions invalides pour dessin droite: y={section_y}, h={height}")
                        section_y += height

            # Mettre à jour la position verticale
            y_pos_right += section_height

    # Sauvegarder les images si nécessaire
    if output_dir and image_name:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/{image_name}_couleurs_globales.jpg", cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/{image_name}_couleurs_split.jpg", cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR))

    return global_img, split_img


# Fonction auxiliaire nécessaire pour extract_dominant_colors_direct
def extract_dominant_colors_direct(image, boolean_mask, n_colors=3):
    """
    Extraire les couleurs dominantes pour une région définie par un masque booléen

    Args:
        image: Image originale (RGB)
        boolean_mask: Masque booléen indiquant les pixels à considérer
        n_colors: Nombre de couleurs dominantes à extraire

    Returns:
        colors: Tableau des couleurs dominantes [R,G,B]
        proportions: Proportions de chaque couleur dominante
    """
    import numpy as np
    from sklearn.cluster import KMeans

    # Redimensionner l'image aux dimensions du masque si nécessaire
    h_mask, w_mask = boolean_mask.shape
    if image.shape[:2] != (h_mask, w_mask):
        import cv2
        image_resized = cv2.resize(image, (w_mask, h_mask))
    else:
        image_resized = image

    # Si aucun pixel n'est sélectionné, retourner None
    if not np.any(boolean_mask):
        return None, None

    # Extraire les pixels de l'image correspondant au masque
    pixels = image_resized[boolean_mask]

    # S'assurer que nous avons assez de pixels pour le clustering
    if len(pixels) < n_colors:
        n_colors = max(1, len(pixels))

    # Appliquer KMeans pour trouver les couleurs dominantes
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
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

# Nouvelle fonction qui extrait directement les couleurs dominantes à partir d'un masque booléen
def extract_dominant_colors_direct(image, boolean_mask, n_colors=3):
    """
    Extraire les couleurs dominantes pour une région définie par un masque booléen

    Args:
        image: Image originale (RGB)
        boolean_mask: Masque booléen indiquant les pixels à considérer
        n_colors: Nombre de couleurs dominantes à extraire

    Returns:
        colors: Tableau des couleurs dominantes [R,G,B]
        proportions: Proportions de chaque couleur dominante
    """
    import numpy as np
    from sklearn.cluster import KMeans

    # Redimensionner l'image aux dimensions du masque si nécessaire
    h_mask, w_mask = boolean_mask.shape
    if image.shape[:2] != (h_mask, w_mask):
        import cv2
        image_resized = cv2.resize(image, (w_mask, h_mask))
    else:
        image_resized = image

    # Si aucun pixel n'est sélectionné, retourner None
    if not np.any(boolean_mask):
        return None, None

    # Extraire les pixels de l'image correspondant au masque
    pixels = image_resized[boolean_mask]

    # S'assurer que nous avons assez de pixels pour le clustering
    if len(pixels) < n_colors:
        n_colors = max(1, len(pixels))

    # Appliquer KMeans pour trouver les couleurs dominantes
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
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


def create_complete_visualization(image, mask, global_img, split_img, cityscapes_colors, cityscapes_classes,
                                  classes_of_interest, colors_per_class, output_dir=None, image_name=None):
    """
    Crée une visualisation complète incluant l'image originale, la segmentation et les images de synthèse

    Args:
        image: Image originale
        mask: Masque de segmentation
        global_img: Image de synthèse avec les couleurs dominantes globales
        split_img: Image de synthèse avec les couleurs dominantes gauche/droite
        cityscapes_colors: Liste des couleurs pour chaque classe
        cityscapes_classes: Dictionnaire des noms de classes
        classes_of_interest: Dictionnaire des classes d'intérêt à analyser
        colors_per_class: Dictionnaire spécifiant le nombre de couleurs à extraire
        output_dir: Répertoire de sortie (optionnel)
        image_name: Nom de base pour les fichiers de sortie (optionnel)

    Returns:
        fig: Figure matplotlib complète
    """
    # Créer la segmentation colorée
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(cityscapes_colors):
        colored_mask[mask == class_id] = color

    # Créer une figure avec un layout fixe
    fig = plt.figure(figsize=(21, 14))

    # 1. Image originale
    ax_img = fig.add_axes([0.05, 0.6, 0.4, 0.3])
    ax_img.imshow(image)
    ax_img.set_title("Image originale", fontsize=14, fontweight='bold')
    ax_img.axis('off')

    # 2. Segmentation
    ax_seg = fig.add_axes([0.55, 0.6, 0.4, 0.3])
    ax_seg.imshow(colored_mask)
    ax_seg.set_title("Segmentation sémantique", fontsize=14, fontweight='bold')
    ax_seg.axis('off')

    # 3. Image de synthèse des couleurs globales
    ax_global = fig.add_axes([0.05, 0.15, 0.4, 0.3])
    ax_global.imshow(global_img)
    ax_global.set_title("Synthèse des couleurs dominantes globales", fontsize=14, fontweight='bold')
    ax_global.axis('off')

    # 4. Image de synthèse des couleurs gauche/droite
    ax_split = fig.add_axes([0.55, 0.15, 0.4, 0.3])
    ax_split.imshow(split_img)
    ax_split.set_title("Synthèse des couleurs dominantes gauche/droite", fontsize=14, fontweight='bold')
    ax_split.axis('off')

    # Créer la légende des classes d'intérêt en haut au centre
    ax_legend = fig.add_axes([0.3, 0.45, 0.4, 0.1])
    ax_legend.axis('off')
    ax_legend.set_title("Classes analysées dans les synthèses", fontsize=14, fontweight='bold')

    # Afficher les classes d'intérêt dans la légende
    x_positions = [0.1, 0.4, 0.7]
    y_pos = 0.6
    rect_height = 0.3
    rect_width = 0.08

    for i, (class_id, class_name) in enumerate(classes_of_interest.items()):
        if class_id in np.unique(mask):  # Vérifier si la classe est présente
            # Positionner au bon endroit selon l'indice
            x = x_positions[i]

            # Récupérer la couleur de cette classe
            color = np.array(cityscapes_colors[class_id]) / 255

            # Dessiner rectangle coloré
            rect = Rectangle((x, y_pos - rect_height), rect_width, rect_height,
                             facecolor=color, edgecolor='black')
            ax_legend.add_patch(rect)

            # Afficher le nom de la classe
            ax_legend.text(x + rect_width + 0.02, y_pos - rect_height / 2,
                           f"{class_name} (Classe {class_id})",
                           va='center', fontsize=10, fontweight='bold')

    # Sauvegarder la figure si nécessaire
    if output_dir and image_name:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(f"{output_dir}/{image_name}_visualisation_complete.png", dpi=300, bbox_inches='tight')

    return fig


def create_dominant_color_images_with_interpolation(image, mask, cityscapes_classes, colors_per_class=None,
                                                    output_dir=None,
                                                    image_name=None, debug=False, interpolation_strength=3):
    """
    Crée deux images de synthèse basées sur les couleurs dominantes des classes d'intérêt avec interpolation
    entre les bandes de couleurs pour des transitions douces.

    Args:
        image: Image originale
        mask: Masque de segmentation
        cityscapes_classes: Dictionnaire des classes
        colors_per_class: Dictionnaire spécifiant le nombre de couleurs à extraire pour chaque classe
        output_dir: Répertoire de sortie (optionnel)
        image_name: Nom de base pour les fichiers de sortie (optionnel)
        debug: Activer le mode debug pour afficher des messages détaillés
        interpolation_strength: Contrôle la taille de la zone d'interpolation (plus élevé = transitions plus douces)

    Returns:
        global_img: Image de synthèse avec les couleurs dominantes globales
        split_img: Image de synthèse avec les couleurs dominantes gauche/droite
    """
    import os
    import numpy as np
    import cv2

    if debug:
        print("Démarrage de la création des images de synthèse avec interpolation")

    # Classes d'intérêt
    classes_of_interest = {
        8: 'végétation',  # vegetation
        9: 'terrain',  # terrain
        10: 'ciel'  # ciel
    }

    # Valeurs par défaut pour le nombre de couleurs si non spécifié
    if colors_per_class is None:
        colors_per_class = {
            8: {'global': 5, 'split': 3},  # végétation: 5 couleurs globales, 3 par côté
            9: {'global': 3, 'split': 3},  # terrain: 3 couleurs globales, 3 par côté
            10: {'global': 2, 'split': 2},  # ciel: 2 couleurs globales, 2 par côté
        }

    # Obtenir les dimensions de l'image et du masque
    h_img, w_img = image.shape[:2]
    h_mask, w_mask = mask.shape

    if debug:
        print(f"Dimensions de l'image originale: {w_img}x{h_img}")
        print(f"Dimensions du masque: {w_mask}x{h_mask}")

    # Calculer les points milieux séparément pour l'image et le masque
    mid_w_img = w_img // 2  # Pour l'image de synthèse
    mid_w_mask = w_mask // 2  # Pour le masque

    if debug:
        print(f"Point milieu image: {mid_w_img}")
        print(f"Point milieu masque: {mid_w_mask}")

    # Trier les classes d'intérêt par ordre vertical (ciel en haut, puis végétation, puis terrain)
    vertical_order = [10, 8, 9]  # [ciel, végétation, terrain]

    # Créer des masques booléens basés sur le masque de segmentation
    mask_left = np.zeros_like(mask, dtype=bool)
    mask_right = np.zeros_like(mask, dtype=bool)

    # Définir les régions gauche et droite en utilisant le point milieu du masque
    mask_left[:, :mid_w_mask] = True
    mask_right[:, mid_w_mask:] = True

    # Identifier les classes présentes et extraire les pixels/couleurs
    pixels_global, pixels_left, pixels_right = {}, {}, {}
    classes_present_global, classes_present_left, classes_present_right = [], [], []
    global_colors, left_colors, right_colors = {}, {}, {}
    global_props, left_props, right_props = {}, {}, {}

    # Identifier les classes présentes et extraire les couleurs
    for class_id in classes_of_interest:
        class_mask = mask == class_id

        # Vérifier si la classe est présente globalement
        if np.any(class_mask):
            classes_present_global.append(class_id)
            pixels_global[class_id] = np.sum(class_mask)

            # Obtenir le nombre de couleurs pour cette classe
            n_colors_global = colors_per_class[class_id]['global'] if class_id in colors_per_class else 5

            # Extraire les couleurs dominantes globales
            colors, props = extract_dominant_colors(image, mask, class_id, n_colors=n_colors_global)
            if colors is not None:
                global_colors[class_id] = colors
                global_props[class_id] = props

        # Vérifier si la classe est présente à gauche
        class_mask_left = np.logical_and(class_mask, mask_left)
        if np.any(class_mask_left):
            classes_present_left.append(class_id)
            pixels_left[class_id] = np.sum(class_mask_left)

        # Vérifier si la classe est présente à droite
        class_mask_right = np.logical_and(class_mask, mask_right)
        if np.any(class_mask_right):
            classes_present_right.append(class_id)
            pixels_right[class_id] = np.sum(class_mask_right)

        # Obtenir le nombre de couleurs pour split
        n_colors_split = colors_per_class[class_id]['split'] if class_id in colors_per_class else 3

        # Extraire directement les couleurs pour chaque côté
        if np.any(class_mask_left):
            left_colors[class_id], left_props[class_id] = extract_dominant_colors_direct(
                image, class_mask_left, n_colors=n_colors_split)

        if np.any(class_mask_right):
            right_colors[class_id], right_props[class_id] = extract_dominant_colors_direct(
                image, class_mask_right, n_colors=n_colors_split)

    # Si aucune classe d'intérêt n'est présente, retourner des images noires
    if len(classes_present_global) == 0:
        if debug:
            print("Aucune classe d'intérêt présente dans l'image")
        global_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        split_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)

        if output_dir and image_name:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(f"{output_dir}/{image_name}_couleurs_globales.jpg", cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{output_dir}/{image_name}_couleurs_split.jpg", cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR))

        return global_img, split_img

    # Trier les classes présentes selon l'ordre vertical
    classes_present_global.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))
    classes_present_left.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))
    classes_present_right.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))

    # Calculer les proportions relatives pour chaque partie
    total_pixels_global = sum(pixels_global.values())
    total_pixels_left = sum(pixels_left.values()) if pixels_left else 0
    total_pixels_right = sum(pixels_right.values()) if pixels_right else 0

    relative_heights_global = {class_id: (pixels_global[class_id] / total_pixels_global) * 100
                               for class_id in classes_present_global}
    relative_heights_left = {class_id: (pixels_left[class_id] / total_pixels_left) * 100
                             for class_id in classes_present_left} if total_pixels_left > 0 else {}
    relative_heights_right = {class_id: (pixels_right[class_id] / total_pixels_right) * 100
                              for class_id in classes_present_right} if total_pixels_right > 0 else {}

    # Créer les images avec interpolation
    global_img = np.zeros((h_img, w_img, 3), dtype=np.float32)
    split_img = np.zeros((h_img, w_img, 3), dtype=np.float32)

    # Fonction utilitaire pour l'interpolation linéaire entre deux couleurs
    def interpolate_colors(color1, color2, num_steps):
        """Crée une interpolation linéaire entre deux couleurs RGB"""
        r = np.linspace(color1[0], color2[0], num_steps)
        g = np.linspace(color1[1], color2[1], num_steps)
        b = np.linspace(color1[2], color2[2], num_steps)
        return np.column_stack((r, g, b))

    # 1. Générer l'image globale avec interpolation
    section_heights_global = []
    remaining_height = h_img

    # Calculer d'abord les hauteurs des sections par classe
    for i, class_id in enumerate(classes_present_global):
        if i == len(classes_present_global) - 1:
            section_heights_global.append(remaining_height)
        else:
            section_height = int((relative_heights_global[class_id] / 100) * h_img)
            section_height = max(1, section_height)
            section_heights_global.append(section_height)
            remaining_height -= section_height

    # Maintenant dessiner l'image avec interpolation
    y_pos = 0
    for i, class_id in enumerate(classes_present_global):
        section_height = section_heights_global[i]

        if class_id in global_colors and global_colors[class_id] is not None:
            colors = global_colors[class_id]
            props = global_props[class_id]

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
                        color_height = max(1, color_height)
                        color_heights.append(color_height)
                        remaining -= color_height

                # Dessiner avec interpolation
                section_y = y_pos
                for j, (color, height) in enumerate(zip(colors, color_heights)):
                    # S'assurer que les dimensions sont valides
                    if height > 0 and section_y < h_img:
                        # Si c'est la dernière couleur de cette classe et il y a une classe suivante
                        if j == len(colors) - 1 and i < len(classes_present_global) - 1:
                            next_class_id = classes_present_global[i + 1]
                            if next_class_id in global_colors and global_colors[next_class_id] is not None:
                                next_color = global_colors[next_class_id][
                                    0]  # Prendre la première couleur de la classe suivante
                                # Calculer la zone d'interpolation (éviter de dépasser les limites)
                                interp_size = max(interpolation_strength, height // 3)

                                # Remplir la partie non-interpolée
                                if height - interp_size > 0:
                                    global_img[section_y:section_y + height - interp_size, :] = color

                                # Créer l'interpolation entre les deux classes
                                if interp_size > 0:
                                    gradient = interpolate_colors(color, next_color, interp_size)
                                    for k in range(interp_size):
                                        if section_y + height - interp_size + k < h_img:
                                            global_img[section_y + height - interp_size + k, :] = gradient[k]
                            else:
                                # Pas de classe suivante valide, remplir normalement
                                global_img[section_y:section_y + height, :] = color
                        # Si ce n'est pas la dernière couleur de cette classe, interpoler avec la couleur suivante
                        elif j < len(colors) - 1:
                            next_color = colors[j + 1]
                            # Calculer la zone d'interpolation
                            interp_size = max(interpolation_strength, height // 3)

                            # Remplir la partie non-interpolée
                            if height - interp_size > 0:
                                global_img[section_y:section_y + height - interp_size, :] = color

                            # Créer l'interpolation
                            if interp_size > 0:
                                gradient = interpolate_colors(color, next_color, interp_size)
                                for k in range(interp_size):
                                    if section_y + height - interp_size + k < h_img:
                                        global_img[section_y + height - interp_size + k, :] = gradient[k]
                        else:
                            # Dernière couleur et dernière classe, remplir normalement
                            end_y = min(section_y + height, h_img)
                            global_img[section_y:end_y, :] = color

                    section_y += height

        # Mettre à jour la position verticale
        y_pos += section_height

    # 2. Générer l'image split avec interpolation (partie gauche)
    if classes_present_left:
        section_heights_left = []
        remaining_height = h_img

        for i, class_id in enumerate(classes_present_left):
            if i == len(classes_present_left) - 1:
                section_heights_left.append(remaining_height)
            else:
                section_height = int((relative_heights_left[class_id] / 100) * h_img)
                section_height = max(1, section_height)
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
                            color_height = max(1, color_height)
                            color_heights.append(color_height)
                            remaining -= color_height

                    # Dessiner avec interpolation
                    section_y = y_pos_left
                    for j, (color, height) in enumerate(zip(colors, color_heights)):
                        if height > 0 and section_y < h_img:
                            # Si c'est la dernière couleur de cette classe et il y a une classe suivante
                            if j == len(colors) - 1 and i < len(classes_present_left) - 1:
                                next_class_id = classes_present_left[i + 1]
                                if next_class_id in left_colors and left_colors[next_class_id] is not None:
                                    next_color = left_colors[next_class_id][0]
                                    interp_size = max(interpolation_strength, height // 3)

                                    if height - interp_size > 0:
                                        split_img[section_y:section_y + height - interp_size, :mid_w_img] = color

                                    if interp_size > 0:
                                        gradient = interpolate_colors(color, next_color, interp_size)
                                        for k in range(interp_size):
                                            if section_y + height - interp_size + k < h_img:
                                                split_img[section_y + height - interp_size + k, :mid_w_img] = gradient[
                                                    k]
                                else:
                                    end_y = min(section_y + height, h_img)
                                    split_img[section_y:end_y, :mid_w_img] = color
                            # Si ce n'est pas la dernière couleur, interpoler avec la suivante
                            elif j < len(colors) - 1:
                                next_color = colors[j + 1]
                                interp_size = max(interpolation_strength, height // 3)

                                if height - interp_size > 0:
                                    split_img[section_y:section_y + height - interp_size, :mid_w_img] = color

                                if interp_size > 0:
                                    gradient = interpolate_colors(color, next_color, interp_size)
                                    for k in range(interp_size):
                                        if section_y + height - interp_size + k < h_img:
                                            split_img[section_y + height - interp_size + k, :mid_w_img] = gradient[k]
                            else:
                                end_y = min(section_y + height, h_img)
                                split_img[section_y:end_y, :mid_w_img] = color

                        section_y += height

            y_pos_left += section_height

    # 3. Générer l'image split avec interpolation (partie droite)
    if classes_present_right:
        section_heights_right = []
        remaining_height = h_img

        for i, class_id in enumerate(classes_present_right):
            if i == len(classes_present_right) - 1:
                section_heights_right.append(remaining_height)
            else:
                section_height = int((relative_heights_right[class_id] / 100) * h_img)
                section_height = max(1, section_height)
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
                            color_height = max(1, color_height)
                            color_heights.append(color_height)
                            remaining -= color_height

                    # Dessiner avec interpolation
                    section_y = y_pos_right
                    for j, (color, height) in enumerate(zip(colors, color_heights)):
                        if height > 0 and section_y < h_img:
                            # Si c'est la dernière couleur de cette classe et il y a une classe suivante
                            if j == len(colors) - 1 and i < len(classes_present_right) - 1:
                                next_class_id = classes_present_right[i + 1]
                                if next_class_id in right_colors and right_colors[next_class_id] is not None:
                                    next_color = right_colors[next_class_id][0]
                                    interp_size = max(interpolation_strength, height // 3)

                                    if height - interp_size > 0:
                                        split_img[section_y:section_y + height - interp_size, mid_w_img:] = color

                                    if interp_size > 0:
                                        gradient = interpolate_colors(color, next_color, interp_size)
                                        for k in range(interp_size):
                                            if section_y + height - interp_size + k < h_img:
                                                split_img[section_y + height - interp_size + k, mid_w_img:] = gradient[
                                                    k]
                                else:
                                    end_y = min(section_y + height, h_img)
                                    split_img[section_y:end_y, mid_w_img:] = color
                            # Si ce n'est pas la dernière couleur, interpoler avec la suivante
                            elif j < len(colors) - 1:
                                next_color = colors[j + 1]
                                interp_size = max(interpolation_strength, height // 3)

                                if height - interp_size > 0:
                                    split_img[section_y:section_y + height - interp_size, mid_w_img:] = color

                                if interp_size > 0:
                                    gradient = interpolate_colors(color, next_color, interp_size)
                                    for k in range(interp_size):
                                        if section_y + height - interp_size + k < h_img:
                                            split_img[section_y + height - interp_size + k, mid_w_img:] = gradient[k]
                            else:
                                end_y = min(section_y + height, h_img)
                                split_img[section_y:end_y, mid_w_img:] = color

                        section_y += height

            y_pos_right += section_height

    # Convertir en uint8 pour l'affichage et la sauvegarde
    global_img = np.clip(global_img, 0, 255).astype(np.uint8)
    split_img = np.clip(split_img, 0, 255).astype(np.uint8)

    # Sauvegarder les images si nécessaire
    if output_dir and image_name:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/{image_name}_couleurs_globales.jpg", cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/{image_name}_couleurs_split.jpg", cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR))

    return global_img, split_img


import numpy as np
import cv2
from sklearn.cluster import KMeans


def extract_dominant_colors_with_position(image, mask, class_id, n_colors=5):
    """
    Extraire les couleurs dominantes pour une classe en gardant l'information de position verticale
    """
    # Redimensionner l'image aux dimensions du masque
    h_mask, w_mask = mask.shape
    image_resized = cv2.resize(image, (w_mask, h_mask))

    # Créer un masque booléen pour la classe
    class_mask = mask == class_id

    # Si aucun pixel n'appartient à cette classe, retourner None
    if not np.any(class_mask):
        return None, None

    # Extraire les pixels de l'image correspondant à la classe et leurs positions
    pixels = []
    positions_y = []
    for y in range(h_mask):
        for x in range(w_mask):
            if class_mask[y, x]:
                pixels.append(image_resized[y, x])
                positions_y.append(y)

    pixels = np.array(pixels)
    positions_y = np.array(positions_y)

    # Éviter une erreur si le nombre de pixels est inférieur à n_colors
    if len(pixels) < n_colors:
        n_colors = max(1, len(pixels))

    # Appliquer KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
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


def extract_dominant_colors_split_with_position(image, mask, class_id, n_colors=3):
    """
    Extraire les couleurs dominantes pour chaque moitié de l'image en conservant l'information de position
    """
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

    left_colors = left_proportions = right_colors = right_proportions = None

    # Traiter la partie gauche si elle contient des pixels
    if np.any(left_mask):
        left_colors, left_proportions = extract_dominant_colors_direct_with_position(
            image, left_mask, n_colors=n_colors)

    # Traiter la partie droite si elle contient des pixels
    if np.any(right_mask):
        right_colors, right_proportions = extract_dominant_colors_direct_with_position(
            image, right_mask, n_colors=n_colors)

    return left_colors, left_proportions, right_colors, right_proportions


def extract_dominant_colors_direct_with_position(image, boolean_mask, n_colors=3):
    """
    Extraire les couleurs dominantes pour une région définie par un masque booléen
    en conservant l'information de position verticale
    """
    # Redimensionner l'image aux dimensions du masque si nécessaire
    h_mask, w_mask = boolean_mask.shape
    if image.shape[:2] != (h_mask, w_mask):
        image_resized = cv2.resize(image, (w_mask, h_mask))
    else:
        image_resized = image

    # Si aucun pixel n'est sélectionné, retourner None
    if not np.any(boolean_mask):
        return None, None

    # Extraire les pixels de l'image correspondant au masque et leurs positions
    pixels = []
    positions_y = []
    for y in range(h_mask):
        for x in range(w_mask):
            if boolean_mask[y, x]:
                pixels.append(image_resized[y, x])
                positions_y.append(y)

    pixels = np.array(pixels)
    positions_y = np.array(positions_y)

    # S'assurer que nous avons assez de pixels pour le clustering
    if len(pixels) < n_colors:
        n_colors = max(1, len(pixels))

    # Appliquer KMeans pour trouver les couleurs dominantes
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
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


# Modification de la fonction create_dominant_color_images pour utiliser les nouvelles fonctions
def create_dominant_color_images_with_position(image, mask, cityscapes_classes, colors_per_class=None, output_dir=None,
                                               image_name=None, debug=False):
    """
    Crée deux images de synthèse basées sur les couleurs dominantes des classes d'intérêt
    en tenant compte de la position verticale des couleurs
    """
    import os

    if debug:
        print("Démarrage de la création des images de synthèse basées sur la position")

    # Classes d'intérêt
    classes_of_interest = {
        8: 'végétation',  # vegetation
        9: 'terrain',  # terrain
        10: 'ciel'  # ciel
    }

    # Valeurs par défaut pour le nombre de couleurs si non spécifié
    if colors_per_class is None:
        colors_per_class = {
            8: {'global': 5, 'split': 3},  # végétation: 5 couleurs globales, 3 par côté
            9: {'global': 3, 'split': 3},  # terrain: 3 couleurs globales, 3 par côté
            10: {'global': 2, 'split': 2},  # ciel: 2 couleurs globales, 2 par côté
        }

    # Obtenir les dimensions de l'image et du masque
    h_img, w_img = image.shape[:2]
    h_mask, w_mask = mask.shape

    # Calculer les points milieux séparément pour l'image et le masque
    mid_w_img = w_img // 2  # Pour l'image de synthèse
    mid_w_mask = w_mask // 2  # Pour le masque

    # Trier les classes d'intérêt par ordre vertical (ciel en haut, puis végétation, puis terrain)
    vertical_order = [10, 8, 9]  # [ciel, végétation, terrain]

    # Créer des masques booléens basés sur le masque de segmentation
    mask_left = np.zeros_like(mask, dtype=bool)
    mask_right = np.zeros_like(mask, dtype=bool)

    # Définir les régions gauche et droite en utilisant le point milieu du masque
    mask_left[:, :mid_w_mask] = True
    mask_right[:, mid_w_mask:] = True

    # Identifier les classes présentes et extraire les pixels/couleurs
    pixels_global, pixels_left, pixels_right = {}, {}, {}
    classes_present_global, classes_present_left, classes_present_right = [], [], []
    global_colors, left_colors, right_colors = {}, {}, {}
    global_props, left_props, right_props = {}, {}, {}

    # Identifier les classes présentes et extraire les couleurs
    for class_id in classes_of_interest:
        class_mask = mask == class_id

        # Vérifier si la classe est présente globalement
        if np.any(class_mask):
            classes_present_global.append(class_id)
            pixels_global[class_id] = np.sum(class_mask)

            # Obtenir le nombre de couleurs pour cette classe
            n_colors_global = colors_per_class[class_id]['global'] if class_id in colors_per_class else 5

            # Extraire les couleurs dominantes globales avec position
            colors, props = extract_dominant_colors_with_position(image, mask, class_id, n_colors=n_colors_global)
            if colors is not None:
                global_colors[class_id] = colors
                global_props[class_id] = props

        # Vérifier si la classe est présente à gauche
        class_mask_left = np.logical_and(class_mask, mask_left)
        if np.any(class_mask_left):
            classes_present_left.append(class_id)
            pixels_left[class_id] = np.sum(class_mask_left)

        # Vérifier si la classe est présente à droite
        class_mask_right = np.logical_and(class_mask, mask_right)
        if np.any(class_mask_right):
            classes_present_right.append(class_id)
            pixels_right[class_id] = np.sum(class_mask_right)

        # Obtenir le nombre de couleurs pour split
        n_colors_split = colors_per_class[class_id]['split'] if class_id in colors_per_class else 3

        # Extraire directement les couleurs pour chaque côté avec position
        if np.any(class_mask_left):
            left_colors[class_id], left_props[class_id] = extract_dominant_colors_direct_with_position(
                image, class_mask_left, n_colors=n_colors_split)

        if np.any(class_mask_right):
            right_colors[class_id], right_props[class_id] = extract_dominant_colors_direct_with_position(
                image, class_mask_right, n_colors=n_colors_split)

    # Si aucune classe d'intérêt n'est présente, retourner des images noires
    if len(classes_present_global) == 0:
        if debug:
            print("Aucune classe d'intérêt présente dans l'image")
        global_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        split_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)

        if output_dir and image_name:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(f"{output_dir}/{image_name}_couleurs_globales.jpg", cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{output_dir}/{image_name}_couleurs_split.jpg", cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR))

        return global_img, split_img

    # Trier les classes présentes selon l'ordre vertical
    classes_present_global.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))
    classes_present_left.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))
    classes_present_right.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))

    # Calculer les proportions relatives pour chaque partie
    total_pixels_global = sum(pixels_global.values())
    total_pixels_left = sum(pixels_left.values()) if pixels_left else 0
    total_pixels_right = sum(pixels_right.values()) if pixels_right else 0

    relative_heights_global = {class_id: (pixels_global[class_id] / total_pixels_global) * 100
                               for class_id in classes_present_global}
    relative_heights_left = {class_id: (pixels_left[class_id] / total_pixels_left) * 100
                             for class_id in classes_present_left} if total_pixels_left > 0 else {}
    relative_heights_right = {class_id: (pixels_right[class_id] / total_pixels_right) * 100
                              for class_id in classes_present_right} if total_pixels_right > 0 else {}

    # Créer les images de synthèse
    global_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    split_img = np.zeros((h_img, w_img, 3), dtype=np.uint8)

    # 1. Générer l'image globale
    section_heights_global = []
    remaining_height = h_img

    for i, class_id in enumerate(classes_present_global):
        if i == len(classes_present_global) - 1:
            section_heights_global.append(remaining_height)
        else:
            section_height = int((relative_heights_global[class_id] / 100) * h_img)
            section_height = max(1, section_height)
            section_heights_global.append(section_height)
            remaining_height -= section_height

    y_pos = 0
    for i, class_id in enumerate(classes_present_global):
        section_height = section_heights_global[i]

        if class_id in global_colors and global_colors[class_id] is not None:
            colors = global_colors[class_id]
            props = global_props[class_id]

            # Normaliser les proportions
            total_prop = sum(props)
            if total_prop > 0:
                normalized_props = props / total_prop

                # Calculer les hauteurs de chaque bande de couleur
                color_heights = []
                remaining = section_height

                for j, prop in enumerate(normalized_props):
                    if j == len(normalized_props) - 1:
                        color_heights.append(remaining)
                    else:
                        color_height = int(prop * section_height)
                        color_height = max(1, color_height)
                        color_heights.append(color_height)
                        remaining -= color_height

                # Dessiner chaque bande de couleur
                section_y = y_pos
                for j, (color, height) in enumerate(zip(colors, color_heights)):
                    if height > 0 and section_y < h_img:
                        color_uint8 = np.clip(color, 0, 255).astype(np.uint8)
                        end_y = min(section_y + height, h_img)
                        global_img[section_y:end_y, :] = color_uint8
                    section_y += height

        y_pos += section_height

    # 2. PARTIE GAUCHE
    if classes_present_left:
        section_heights_left = []
        remaining_height = h_img

        for i, class_id in enumerate(classes_present_left):
            if i == len(classes_present_left) - 1:
                section_heights_left.append(remaining_height)
            else:
                section_height = int((relative_heights_left[class_id] / 100) * h_img)
                section_height = max(1, section_height)
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

                    # Calculer les hauteurs de chaque bande de couleur
                    color_heights = []
                    remaining = section_height

                    for j, prop in enumerate(normalized_props):
                        if j == len(normalized_props) - 1:
                            color_heights.append(remaining)
                        else:
                            color_height = int(prop * section_height)
                            color_height = max(1, color_height)
                            color_heights.append(color_height)
                            remaining -= color_height

                    # Dessiner chaque bande de couleur
                    section_y = y_pos_left
                    for j, (color, height) in enumerate(zip(colors, color_heights)):
                        if height > 0 and section_y < h_img:
                            color_uint8 = np.clip(color, 0, 255).astype(np.uint8)
                            end_y = min(section_y + height, h_img)
                            split_img[section_y:end_y, :mid_w_img] = color_uint8
                        section_y += height

            y_pos_left += section_height

    # 3. PARTIE DROITE
    if classes_present_right:
        section_heights_right = []
        remaining_height = h_img

        for i, class_id in enumerate(classes_present_right):
            if i == len(classes_present_right) - 1:
                section_heights_right.append(remaining_height)
            else:
                section_height = int((relative_heights_right[class_id] / 100) * h_img)
                section_height = max(1, section_height)
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

                    # Calculer les hauteurs de chaque bande de couleur
                    color_heights = []
                    remaining = section_height

                    for j, prop in enumerate(normalized_props):
                        if j == len(normalized_props) - 1:
                            color_heights.append(remaining)
                        else:
                            color_height = int(prop * section_height)
                            color_height = max(1, color_height)
                            color_heights.append(color_height)
                            remaining -= color_height

                    # Dessiner chaque bande de couleur
                    section_y = y_pos_right
                    for j, (color, height) in enumerate(zip(colors, color_heights)):
                        if height > 0 and section_y < h_img:
                            color_uint8 = np.clip(color, 0, 255).astype(np.uint8)
                            end_y = min(section_y + height, h_img)
                            split_img[section_y:end_y, mid_w_img:] = color_uint8
                        section_y += height

            y_pos_right += section_height

    # Sauvegarder les images si nécessaire
    if output_dir and image_name:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/{image_name}_couleurs_globales.jpg", cv2.cvtColor(global_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/{image_name}_couleurs_split.jpg", cv2.cvtColor(split_img, cv2.COLOR_RGB2BGR))

    return global_img, split_img


# Version avec interpolation qui tient compte de la position verticale
def create_dominant_color_images_with_interpolation_and_position(image, mask, cityscapes_classes, colors_per_class=None,
                                                                 output_dir=None, image_name=None, debug=False,
                                                                 interpolation_strength=3):
    """
    Crée deux images de synthèse basées sur les couleurs dominantes des classes d'intérêt avec interpolation
    entre les bandes de couleurs pour des transitions douces, en tenant compte de la position verticale.
    """
    import os

    if debug:
        print("Démarrage de la création des images de synthèse avec interpolation basée sur la position")

    # Classes d'intérêt
    classes_of_interest = {
        8: 'végétation',  # vegetation
        9: 'terrain',  # terrain
        10: 'ciel'  # ciel
    }

    # Valeurs par défaut pour le nombre de couleurs si non spécifié
    if colors_per_class is None:
        colors_per_class = {
            8: {'global': 5, 'split': 3},  # végétation: 5 couleurs globales, 3 par côté
            9: {'global': 3, 'split': 3},  # terrain: 3 couleurs globales, 3 par côté
            10: {'global': 2, 'split': 2},  # ciel: 2 couleurs globales, 2 par côté
        }

    # Obtenir les dimensions de l'image et du masque
    h_img, w_img = image.shape[:2]
    h_mask, w_mask = mask.shape

    # Calculer les points milieux séparément pour l'image et le masque
    mid_w_img = w_img // 2  # Pour l'image de synthèse
    mid_w_mask = w_mask // 2  # Pour le masque

    # Trier les classes d'intérêt par ordre vertical (ciel en haut, puis végétation, puis terrain)
    vertical_order = [10, 8, 9]  # [ciel, végétation, terrain]

    # Créer des masques booléens basés sur le masque de segmentation
    mask_left = np.zeros_like(mask, dtype=bool)
    mask_right = np.zeros_like(mask, dtype=bool)

    # Définir les régions gauche et droite en utilisant le point milieu du masque
    mask_left[:, :mid_w_mask] = True
    mask_right[:, mid_w_mask:] = True

    # Identifier les classes présentes et extraire les pixels/couleurs
    pixels_global, pixels_left, pixels_right = {}, {}, {}
    classes_present_global, classes_present_left, classes_present_right = [], [], []
    global_colors, left_colors, right_colors = {}, {}, {}
    global_props, left_props, right_props = {}, {}, {}

    # Identifier les classes présentes et extraire les couleurs
    for class_id in classes_of_interest:
        class_mask = mask == class_id

        # Vérifier si la classe est présente globalement
        if np.any(class_mask):
            classes_present_global.append(class_id)
            pixels_global[class_id] = np.sum(class_mask)

            # Obtenir le nombre de couleurs pour cette classe
            n_colors_global = colors_per_class[class_id]['global'] if class_id in colors_per_class else 5

            # Extraire les couleurs dominantes globales avec position
            colors, props = extract_dominant_colors_with_position(image, mask, class_id, n_colors=n_colors_global)
            if colors is not None:
                global_colors[class_id] = colors
                global_props[class_id] = props

        # Vérifier si la classe est présente à gauche
        class_mask_left = np.logical_and(class_mask, mask_left)
        if np.any(class_mask_left):
            classes_present_left.append(class_id)
            pixels_left[class_id] = np.sum(class_mask_left)

        # Vérifier si la classe est présente à droite
        class_mask_right = np.logical_and(class_mask, mask_right)
        if np.any(class_mask_right):
            classes_present_right.append(class_id)
            pixels_right[class_id] = np.sum(class_mask_right)

        # Obtenir le nombre de couleurs pour split
        n_colors_split = colors_per_class[class_id]['split'] if class_id in colors_per_class else 3

        # Extraire directement les couleurs pour chaque côté avec position
        if np.any(class_mask_left):
            left_colors[class_id], left_props[class_id] = extract_dominant_colors_direct_with_position(
                image, class_mask_left, n_colors=n_colors_split)

        if np.any(class_mask_right):
            right_colors[class_id], right_props[class_id] = extract_dominant_colors_direct_with_position(
                image, class_mask_right, n_colors=n_colors_split)

    # Si aucune classe d'intérêt n'est présente, retourner des images noires
    if len(classes_present_global) == 0:
        if debug:
            print("Aucune classe d'intérêt présente dans l'image")
        global_img = np.zeros((h_img, w_img, 3), dtype=np.float32)
        split_img = np.zeros((h_img, w_img, 3), dtype=np.float32)

        if output_dir and image_name:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(f"{output_dir}/{image_name}_couleurs_globales.jpg",
                        cv2.cvtColor(global_img.astype(np.uint8), cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"{output_dir}/{image_name}_couleurs_split.jpg",
                        cv2.cvtColor(split_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

        return global_img.astype(np.uint8), split_img.astype(np.uint8)

    # Trier les classes présentes selon l'ordre vertical
    classes_present_global.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))
    classes_present_left.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))
    classes_present_right.sort(key=lambda x: vertical_order.index(x) if x in vertical_order else len(vertical_order))

    # Calculer les proportions relatives pour chaque partie
    total_pixels_global = sum(pixels_global.values())
    total_pixels_left = sum(pixels_left.values()) if pixels_left else 0
    total_pixels_right = sum(pixels_right.values()) if pixels_right else 0

    relative_heights_global = {class_id: (pixels_global[class_id] / total_pixels_global) * 100
                               for class_id in classes_present_global}
    relative_heights_left = {class_id: (pixels_left[class_id] / total_pixels_left) * 100
                             for class_id in classes_present_left} if total_pixels_left > 0 else {}
    relative_heights_right = {class_id: (pixels_right[class_id] / total_pixels_right) * 100
                              for class_id in classes_present_right} if total_pixels_right > 0 else {}

    # Créer les images avec interpolation
    global_img = np.zeros((h_img, w_img, 3), dtype=np.float32)
    split_img = np.zeros((h_img, w_img, 3), dtype=np.float32)

    # Fonction utilitaire pour l'interpolation linéaire entre deux couleurs
    def interpolate_colors(color1, color2, num_steps):
        """Crée une interpolation linéaire entre deux couleurs RGB"""
        r = np.linspace(color1[0], color2[0], num_steps)
        g = np.linspace(color1[1], color2[1], num_steps)
        b = np.linspace(color1[2], color2[2], num_steps)
        return np.column_stack((r, g, b))

    # 1. Générer l'image globale avec interpolation
    section_heights_global = []
    remaining_height = h_img

    # Calculer d'abord les hauteurs des sections par classe
    for i, class_id in enumerate(classes_present_global):
        if i == len(classes_present_global) - 1:
            section_heights_global.append(remaining_height)
        else:
            section_height = int((relative_heights_global[class_id] / 100) * h_img)
            section_height = max(1, section_height)
            section_heights_global.append(section_height)
            remaining_height -= section_height

    # Maintenant dessiner l'image avec interpolation
    y_pos = 0
    for i, class_id in enumerate(classes_present_global):
        section_height = section_heights_global[i]

        if class_id in global_colors and global_colors[class_id] is not None:
            colors = global_colors[class_id]
            props = global_props[class_id]

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
                        color_height = max(1, color_height)
                        color_heights.append(color_height)
                        remaining -= color_height

                # Dessiner avec interpolation
                section_y = y_pos
                for j, (color, height) in enumerate(zip(colors, color_heights)):
                    # S'assurer que les dimensions sont valides
                    if height > 0 and section_y < h_img:
                        # Si c'est la dernière couleur de cette classe et il y a une classe suivante
                        if j == len(colors) - 1 and i < len(classes_present_global) - 1:
                            next_class_id = classes_present_global[i + 1]
                            if next_class_id in global_colors and global_colors[next_class_id] is not None:
                                next_color = global_colors[next_class_id][
                                    0]  # Prendre la première couleur de la classe suivante
                                # Calculer la zone d'interpolation (éviter de dépasser les limites)
                                interp_size = max(interpolation_strength, height // 3)

                                # Remplir la partie non-interpolée
                                if height - interp_size > 0:
                                    global_img[section_y:section_y + height - interp_size, :] = color

                                # Créer l'interpolation entre les deux classes
                                if interp_size > 0:
                                    gradient = interpolate_colors(color, next_color, interp_size)
                                    for k in range(interp_size):
                                        if section_y + height - interp_size + k < h_img:
                                            global_img[section_y + height - interp_size + k, :] = gradient[k]
                            else:
                                # Pas de classe suivante valide, remplir normalement
                                global_img[section_y:section_y + height, :] = color
                        # Si ce n'est pas la dernière couleur de cette classe, interpoler avec la couleur suivante
                        elif j < len(colors) - 1:
                            next_color = colors[j + 1]
                            # Calculer la zone d'interpolation
                            interp_size = max(interpolation_strength, height // 3)

                            # Remplir la partie non-interpolée
                            if height - interp_size > 0:
                                global_img[section_y:section_y + height - interp_size, :] = color

                            # Créer l'interpolation
                            if interp_size > 0:
                                gradient = interpolate_colors(color, next_color, interp_size)
                                for k in range(interp_size):
                                    if section_y + height - interp_size + k < h_img:
                                        global_img[section_y + height - interp_size + k, :] = gradient[k]
                        else:
                            # Dernière couleur et dernière classe, remplir normalement
                            end_y = min(section_y + height, h_img)
                            global_img[section_y:end_y, :] = color

                    section_y += height

        # Mettre à jour la position verticale
        y_pos += section_height

    # 2. Générer l'image split avec interpolation (partie gauche)
    if classes_present_left:
        section_heights_left = []
        remaining_height = h_img

        for i, class_id in enumerate(classes_present_left):
            if i == len(classes_present_left) - 1:
                section_heights_left.append(remaining_height)
            else:
                section_height = int((relative_heights_left[class_id] / 100) * h_img)
                section_height = max(1, section_height)
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
                            color_height = max(1, color_height)
                            color_heights.append(color_height)
                            remaining -= color_height

                    # Dessiner avec interpolation
                    section_y = y_pos_left
                    for j, (color, height) in enumerate(zip(colors, color_heights)):
                        if height > 0 and section_y < h_img:
                            # Si c'est la dernière couleur de cette classe et il y a une classe suivante
                            if j == len(colors) - 1 and i < len(classes_present_left) - 1:
                                next_class_id = classes_present_left[i + 1]
                                if next_class_id in left_colors and left_colors[next_class_id] is not None:
                                    next_color = left_colors[next_class_id][0]
                                    interp_size = max(interpolation_strength, height // 3)

                                    if height - interp_size > 0:
                                        split_img[section_y:section_y + height - interp_size, :mid_w_img] = color

                                    if interp_size > 0:
                                        gradient = interpolate_colors(color, next_color, interp_size)
                                        for k in range(interp_size):
                                            if section_y + height - interp_size + k < h_img:
                                                split_img[section_y + height - interp_size + k, :mid_w_img] = gradient[
                                                    k]
                                else:
                                    end_y = min(section_y + height, h_img)
                                    split_img[section_y:end_y, :mid_w_img] = color
                            # Si ce n'est pas la dernière couleur, interpoler avec la suivante
                            elif j < len(colors) - 1:
                                next_color = colors[j + 1]
                                interp_size = max(interpolation_strength, height // 3)

                                if height - interp_size > 0:
                                    split_img[section_y:section_y + height - interp_size, :mid_w_img] = color

                                if interp_size > 0:
                                    gradient = interpolate_colors(color, next_color, interp_size)
                                    for k in range(interp_size):
                                        if section_y + height - interp_size + k < h_img:
                                            split_img[section_y + height - interp_size + k, :mid_w_img] = gradient[k]
                            else:
                                end_y = min(section_y + height, h_img)
                                split_img[section_y:end_y, :mid_w_img] = color

                        section_y += height

            y_pos_left += section_height

    # 3. Générer l'image split avec interpolation (partie droite)
    if classes_present_right:
        section_heights_right = []
        remaining_height = h_img

        for i, class_id in enumerate(classes_present_right):
            if i == len(classes_present_right) - 1:
                section_heights_right.append(remaining_height)
            else:
                section_height = int((relative_heights_right[class_id] / 100) * h_img)
                section_height = max(1, section_height)
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
                            color_height = max(1, color_height)
                            color_heights.append(color_height)
                            remaining -= color_height

                    # Dessiner avec interpolation
                    section_y = y_pos_right
                    for j, (color, height) in enumerate(zip(colors, color_heights)):
                        if height > 0 and section_y < h_img:
                            # Si c'est la dernière couleur de cette classe et il y a une classe suivante
                            if j == len(colors) - 1 and i < len(classes_present_right) - 1:
                                next_class_id = classes_present_right[i + 1]
                                if next_class_id in right_colors and right_colors[next_class_id] is not None:
                                    next_color = right_colors[next_class_id][0]
                                    interp_size = max(interpolation_strength, height // 3)

                                    if height - interp_size > 0:
                                        split_img[section_y:section_y + height - interp_size, mid_w_img:] = color

                                    if interp_size > 0:
                                        gradient = interpolate_colors(color, next_color, interp_size)
                                        for k in range(interp_size):
                                            if section_y + height - interp_size + k < h_img:
                                                split_img[section_y + height - interp_size + k, mid_w_img:] = gradient[
                                                    k]
                                else:
                                    end_y = min(section_y + height, h_img)
                                    split_img[section_y:end_y, mid_w_img:] = color
                            # Si ce n'est pas la dernière couleur, interpoler avec la suivante
                            elif j < len(colors) - 1:
                                next_color = colors[j + 1]
                                interp_size = max(interpolation_strength, height // 3)

                                if height - interp_size > 0:
                                    split_img[section_y:section_y + height - interp_size, mid_w_img:] = color

                                if interp_size > 0:
                                    gradient = interpolate_colors(color, next_color, interp_size)
                                    for k in range(interp_size):
                                        if section_y + height - interp_size + k < h_img:
                                            split_img[section_y + height - interp_size + k, mid_w_img:] = gradient[k]
                            else:
                                end_y = min(section_y + height, h_img)
                                split_img[section_y:end_y, mid_w_img:] = color

                        section_y += height

            y_pos_right += section_height

    # Convertir en uint8 pour l'affichage et la sauvegarde
    global_img_uint8 = np.clip(global_img, 0, 255).astype(np.uint8)
    split_img_uint8 = np.clip(split_img, 0, 255).astype(np.uint8)

    # Sauvegarder les images si nécessaire
    if output_dir and image_name:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(f"{output_dir}/{image_name}_couleurs_globales.jpg",
                    cv2.cvtColor(global_img_uint8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_dir}/{image_name}_couleurs_split.jpg", cv2.cvtColor(split_img_uint8, cv2.COLOR_RGB2BGR))

    return global_img_uint8, split_img_uint8