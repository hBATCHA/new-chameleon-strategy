# 🦎 New Chameleon Strategy

**New Chameleon Strategy** est une plateforme modulaire de **segmentation sémantique** d'images et de vidéos, basée sur **DeepLabV3+ (MobileNet, Cityscapes)**. Le projet vise à analyser les **couleurs dominantes** des classes segmentées pour des cas d'usage embarqués, industriels ou visuels avancés.

> 📌 Projet développé par [Hashif Batcha](https://github.com/hBATCHA)  
> 🧪 Initialement focalisé sur l'image, puis étendu à la vidéo.

---

## 📁 Structure du projet

```
new-chameleon-strategy/
├── chameleon/
│   ├── basic_segmentation.py               
│   ├── analyze_image_colors.py             
│   ├── analyze_image_colors_split.py       
│   ├── analyze_image.py                    
│   ├── color_analysis_utils.py             
│   └── visualization_utils.py              
│
├── video/
│   ├── analyze_video_frames.py                  # 🎥 Vidéo V1 – traitement vidéo frame par frame
│   ├── analyze_video_frames_skipping.py         # 🎥 Vidéo V2 – skipping + interpolation (segmentation)
│   └── video_frameskip_interpolator.py          # ⚙️ Outil générique skipping/interpolation (non lié à DeepLab)
│
├── samples/                                
├── video_inputs/                           
├── video_outputs/                          
├── results/                                
├── checkpoints/                            
├── DeepLabV3Plus-Pytorch/                  
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Fonctionnalités

### ✅ Partie Image (Versions V1 → V4)

| Version | Script                          | Fonction principale |
|---------|----------------------------------|---------------------|
| V1      | `basic_segmentation.py`         | Segmentation simple |
| V2      | `analyze_image_colors.py`       | Couleurs dominantes |
| V3      | `analyze_image_colors_split.py` | Split gauche/droite |
| V4      | `analyze_image.py`              | Version finale (export, options) |

---

### 🎥 Partie Vidéo

| Version | Script                              | Fonction principale |
|---------|--------------------------------------|---------------------|
| V1      | `analyze_video_frames.py`           | Segmentation frame par frame |
| V2      | `analyze_video_frames_skipping.py`  | Segmentation 1 sur N + interpolation |
| Tool    | `video_frameskip_interpolator.py`   | ✅ Outil générique skipping + effets visuels |

---

## 📂 Répertoires importants

- `video_inputs/` : vidéos à traiter
- `video_outputs/` : vidéos générées automatiquement (non versionné)
- `results/` : sorties images (non versionné)
- `checkpoints/` : modèles DeepLab pré-entraînés

---

## ⚙️ Prérequis

- Python 3.7+
- CUDA (optionnel)
- Dépendances :

```txt
torch
torchvision
numpy
opencv-python
matplotlib
scikit-learn
```

Installation :

```bash
pip install -r requirements.txt
```

---

## 🚀 Exécution

### 🖼️ Image (version finale) :

```bash
python chameleon/analyze_image.py
```

### 📼 Vidéo – frame par frame :

```bash
python video/analyze_video_frames.py
```

### ⚡ Vidéo – skipping + interpolation :

```bash
python video/analyze_video_frames_skipping.py
```

### 🛠️ Outil générique skipping/interpolation :

```bash
python video/video_frameskip_interpolator.py
```

---

## 📌 Modèle utilisé

- Modèle : **DeepLabV3+ MobileNet**
- Checkpoint : `checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth`
- Dataset : Cityscapes (19 classes)

---

## 👨‍💻 Auteur

Développé par **Hashif Batcha**  
- Projet pédagogique
- Structuré pour évoluer vers une démo embarquée/temps réel

---

## 📄 Licence

Usage expérimental ou personnel autorisé.  
Pour publication ou usage professionnel : contacter l’auteur.

---

## 🧭 TODO

- Version webcam (temps réel)
