# 🦎 New Chameleon Strategy

**New Chameleon Strategy** est une plateforme modulaire de **segmentation sémantique** d'images et de vidéos, basée sur **DeepLabV3+ (MobileNet, Cityscapes)**. Le projet vise à analyser les **couleurs dominantes** des classes segmentées pour des cas d'usage embarqués, industriels ou visuels avancés.

> 📌 Projet développé par [Hashif Batcha](https://github.com/hBATCHA)  
> 🧪 Initialement focalisé sur l'image, puis étendu à la vidéo.

---

## 📁 Structure du projet

```
new-chameleon-strategy/
├── chameleon/
│   ├── basic_segmentation.py               # V1 – segmentation simple
│   ├── analyze_image_colors.py             # V2 – ajout de couleurs dominantes
│   ├── analyze_image_colors_split.py       # V3 – split gauche/droite
│   ├── analyze_image.py                    # ✅ V4 – version finale image
│   ├── color_analysis_utils.py             # Module logique (KMeans, split, etc.)
│   └── visualization_utils.py              # Affichage et visualisation
│
├── video/
│   └── analyze_video_frames.py             # 🎥 V1 – traitement vidéo frame par frame
│
├── samples/                                # Images de test
├── video_inputs/                           # Vidéos d'entrée à traiter
├── video_outputs/                          # (Non versionné) – vidéos générées automatiquement
├── results/                                # (Non versionné) – résultats image (masques, etc.)
├── checkpoints/                            # Modèle DeepLabV3+ pré-entraîné
├── DeepLabV3Plus-Pytorch/                  # Code original de VainF (modifié)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Fonctionnalités

### ✅ Partie Image (Versions V1 → V4)

| Version | Script                          | Fonction principale |
|---------|----------------------------------|---------------------|
| V1      | `basic_segmentation.py`         | Segmentation simple (masques) |
| V2      | `analyze_image_colors.py`       | Ajout des couleurs dominantes par classe |
| V3      | `analyze_image_colors_split.py` | Analyse gauche/droite des classes |
| V4      | `analyze_image.py`              | ✅ Version stable : options avancées, visualisation, export |

---

### 🎥 Partie Vidéo

| Version | Script                        | Fonction principale |
|---------|-------------------------------|---------------------|
| V1      | `analyze_video_frames.py`     | Traitement d'une vidéo frame par frame (offline) |
| V2+     | (à venir)                     | Temps réel (caméra, vidéo live, etc.) |

Les vidéos à traiter doivent être placées dans `video_inputs/`.  
Les résultats sont enregistrés :
- dans `results/` pour les images
- dans `video_outputs/` pour les vidéos

---

## ⚙️ Prérequis

- Python 3.7+
- CUDA (optionnel, recommandé)
- Dépendances :

```txt
torch
torchvision
numpy
opencv-python
matplotlib
scikit-learn
```

📦 Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## 🚀 Exécution

### 🔍 Analyse d'image (version finale) :

```bash
python chameleon/analyze_image.py
```

### 📼 Analyse vidéo (offline, image par image) :

```bash
python video/analyze_video_frames.py
```

---

## 📌 Modèle utilisé

- Architecture : **DeepLabV3+** avec backbone **MobileNet**
- Checkpoint : `checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth`
- Entraînement : dataset **Cityscapes**, 19 classes

---

## 👨‍💻 Auteur

Développé par **Hashif Batcha**  
- Projet pédagogique et expérimental
- Structure pensée pour évoluer vers des systèmes embarqués ou temps réel
- Basé sur [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)

---

## 📄 Licence

Ce projet est open-source à des fins d’expérimentation et de démonstration.  
Pour un usage commercial ou publication, contactez l’auteur.

---

## 🧭 Prochaines étapes (TODO)

- Version webcam (temps réel)
