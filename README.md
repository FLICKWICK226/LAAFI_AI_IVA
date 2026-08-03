# 🔬 LAAFI_AI IVA Engine (Version 2.0)
> **Computer-Aided Detection & Diagnosis (CADe/CADx) for Cervical Cancer Screening using Acetic Acid (VIA)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Albumentations](https://img.shields.io/badge/Augmentation-Albumentations-green.svg)](https://albumentations.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Présentation du Projet

Le moteur IA IVA de **LAAFI_AI** est un système de vision par ordinateur à deux étages (*CADe/CADx*) conçu pour assister les professionnels de santé (sages-femmes, infirmiers) en Afrique subsaharienne lors du dépistage du cancer du col de l'utérus par **Inspection Visuelle à l'Acide Acétique (IVA)**.

Il surmonte les contraintes du terrain (smartphone bas de gamme, flou de bougé, reflets de flash LED, présence de sang ou de glaire) grâce à une architecture robuste et des garde-fous cliniques stricts.

---

## 🏗️ Architecture du Modèle (Pipeline à 2 Étages)

```text
 ┌───────────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────────┐
 │   IMAGE BRUTE SMARTPHONE  │  ────►  │   STAGE 1 : CADe (YOLO)   │  ────►  │   CROP JSC (384x384)      │
 │ (Spéculum, Vagin, Reflets)│         │ Localisation Bounding Box │         │  Élimination > 70% Déchets│
 └───────────────────────────┘         └───────────────────────────┘         └───────────────────────────┘
                                                                                           │
                                                                                           ▼
 ┌───────────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────────┐
 │   EXPOSITION ONNX / API   │  ◄────  │  SEUIL T CALIBRÉ (p>=T)   │  ◄────  │ STAGE 2 : CADx (ConvNeXt) │
 │  (ONNX Runtime / Mobile)  │         │  Sensibilité >= 95.0%     │         │ Multi-head Classification │
 └───────────────────────────┘         └───────────────────────────┘         └───────────────────────────┘
```

1. **Étape 1 (CADe - Localisation ROI)** : Détecteur **YOLOv8-Det / YOLOv11-Det** entraîné pour isoler la **Zone de Jonction Squamo-Columnaire (JSC)** et éliminer plus de 70% de l'arrière-plan parasite (métal du spéculum, parois vaginales).
2. **Étape 2 (CADx - Classification Multi-tâche)** : Backbone **ConvNeXt-Base** ou **Swin Transformer v2** effectuant une double prédiction :
   - **Tâche A (Éligibilité IVA)** : JSC visible (Type 1/2 = Éligible) vs JSC non-visible (Type 3 = Invalide/Endocervical).
   - **Tâche B (Diagnostic Lésionnel)** : Presence de lésion acéto-blanche précancéreuse (VIA Positif vs VIA Négatif).

---

## 🎯 Métriques Cibles (Sécurité Clinique SaMD)

Conformément aux directives internationales de dispositifs médicaux (FDA CADe/CADx et CE MDR Classe IIb) :

| Métrique | Seuil Baseline | Seuil Cible (SOTA) | Justification Clinique |
| :--- | :--- | :--- | :--- |
| **Sensibilité (Recall)** | **$\ge 95.0\%$** | **$97.0\%$** | **Sécurité Patient :** Zéro cancer raté (Faux Négatif pénalisé 5x). |
| **Spécificité** | **$\ge 80.0\%$** | **$85.0\%$** | **Efficacité Système :** Éviter les biopsies inutiles par sur-référence. |
| **Score $F_2$** | **$\ge 0.88$** | **$\ge 0.92$** | Pondération du Recall 2x plus forte que la Précision. |
| **AUC-ROC** | **$\ge 0.90$** | **$\ge 0.94$** | Capacité globale de discrimination binaire. |

---

## 📂 Structure du Répertoire

```text
LAAFI_AI_IVA/
│
├── config/
│   └── config.yaml               # Hyperparamètres, chemins et graine SEED = 42
│
├── notebooks/                    # 📓 NOTEBOOKS COLAB EXÉCUTABLES
│   ├── 01_setup.ipynb            # Setup GPU T4, montage Drive & secrets Kaggle
│   ├── 02_data_preparation.ipynb # Ingestion, Clustering patients & Masques Perlin
│   ├── 03_train.ipynb            # Entraînement Stage 1 + Stage 2 (Focal Loss + AMP)
│   └── 04_eval_inference.ipynb   # Calibrage du seuil T, audit Grad-CAM & Export ONNX
│
├── data/                         # 💾 STOCKAGE DONNÉES (Ignoré par Git)
│   ├── raw/                      # Dataset brut Intel-MobileODT & NIH Cervigram
│   ├── synthetic_masks/          # 1 000 masques Perlin .npy pré-générés hors-ligne
│   └── processed/                # Crops JSC (384x384) et splits par patient
│
├── models/                       # 💾 CHECKPOINTS & EXPORTS (Ignorés par Git)
│   ├── checkpoints/              # Checkpoints intermédiaires (.pt)
│   └── exported/                 # Modèle final quantifié (ONNX / OpenVINO)
│
├── outputs/                      # 📊 RÉSULTATS & RAPPORTS
│   ├── figures/                  # Matrice de confusion, ROC curve, Grad-CAM audit
│   ├── reports/                  # Rapports CSV et JSON
│   └── logs/                     # Logs d'entraînement
│
├── src/                          # 📦 PACKAGE PYTHON MODULAIRE
│   ├── data/                     # Ingestion, clustering patients, dataset PyTorch, augmentations
│   ├── models/                   # Wrappers YOLO-Det (Stage 1) et ConvNeXt/Swin (Stage 2)
│   ├── utils/                    # Seed, Focal Loss, métriques cliniques, Grad-CAM
│   └── train.py                  # Engine de training principal
│
├── .gitignore
├── requirements.txt
├── setup.py                      # Package installation (pip install -e .)
└── README.md
```

---

## 🚀 Installation & Démarrage Rapide

### 1. Clonage du Répertoire
```bash
git clone https://github.com/FLICKWICK226/LAAFI_AI_IVA.git
cd LAAFI_AI_IVA
```

### 2. Installation de l'Environnement Virtuel
```bash
python -m venv .venv
# Sur Linux/macOS :
source .venv/bin/activate
# Sur Windows :
.\.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 3. Exécution du Setup et des Tests
```bash
# Vérification du déterminisme SEED = 42
python src/utils/seed.py

# Pré-génération des 1 000 masques de Perlin hors-ligne (0 ms CPU pendant le training)
python src/data/generate_perlin_masks.py
```

---

## 🛡️ Les 6 Commandements Anti-Pièges Cliniques

1. **Split Patient Strict** : Découpage Train/Val/Test uniquement par `patient_id` (via `GroupKFold`) pour zéro fuite de données.
2. **CADe par Bounding Box** : Crop strict de la JSC via YOLO pour éliminer l'arrière-plan.
3. **Augmentations Biologiques Vectorisées** : Interdiction du Hue Shift agressif (`hue_shift <= 0.05`) pour préserver l'acéto-blanchiment.
4. **Pivot d'Éligibilité & Diagnostic** : Distinction claire entre visibilité de la JSC (Type 1/2 vs 3) et pathologie (VIA Positif vs Négatif).
5. **Focal Loss & Seuil Calibré** : `FocalLoss` ($\gamma=2.0$) et seuil $T$ calibré sur la courbe ROC (pas de $p=0.5$ par défaut).
6. **Audit Visuel Grad-CAM** : Vérification systématique que l'attention du modèle se porte sur l'épithélium et non sur les artefacts.

---

## 📄 Licence
Projet distribué sous la licence [MIT](LICENSE). Développé pour la recherche et l'assistance clinique SaMD.
