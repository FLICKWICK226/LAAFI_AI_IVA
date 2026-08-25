# 🔬 LAAFI_AI IVA Engine (Version 2.0)
> **Computer-Aided Detection & Diagnosis (CADe/CADx) for Cervical Cancer Screening using Acetic Acid (VIA)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![Albumentations](https://img.shields.io/badge/Augmentation-Albumentations-green.svg)](https://albumentations.ai/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Présentation du Projet

Le moteur IA IVA de **LAAFI_AI** est un système de vision par ordinateur à deux étages (*CADe/CADx*) conçu pour assister les professionnels de santé (sages-femmes, infirmiers) en Afrique subsaharienne (Burkina Faso / LMIC) lors du dépistage du cancer du col de l'utérus par **Inspection Visuelle à l'Acide Acétique (IVA/VIA)** selon l'approche *Screen-and-Treat* recommandée par l'OMS.

Il fonctionne **100% hors-ligne sur smartphone Android ARM64**, et résiste aux contraintes du terrain (flou de bougé, reflets du flash LED, mucus, sang) grâce à des garde-fous cliniques stricts.

---

## 🏗️ Architecture du Pipeline (CADe / CADx)

```text
 ┌───────────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────────┐
 │   IMAGE BRUTE SMARTPHONE  │  ────►  │   STAGE 1 : CADe (YOLO)   │  ────►  │   CROP JSC (384x384)      │
 │ (Spéculum, Vagin, Reflets)│         │ Localisation Bounding Box │         │  Élimination > 70% Déchets│
 └───────────────────────────┘         └───────────────────────────┘         └───────────────────────────┘
                                                                                           │
                                                                                           ▼
 ┌───────────────────────────┐         ┌───────────────────────────┐         ┌───────────────────────────┐
 │    DÉPLOIEMENT EDGE AI    │  ◄────  │  SEUIL T CALIBRÉ (p>=T)   │  ◄────  │ STAGE 2 : CADx (Teacher/S)│
 │  (ONNX Runtime / INT8)    │         │  Sensibilité >= 95.0%     │         │ ConvNeXt-Base / MobileNet │
 └───────────────────────────┘         └───────────────────────────┘         └───────────────────────────┘
```

1. **Étape 1 (CADe - Localisation ROI)** : Détecteur **YOLOv8-Det / YOLOv11-Det** isolant la **Zone de Jonction Squamo-Columnaire (JSC)**.
2. **Étape 2 (CADx - Classification Diagnostique)** :
   - **Teacher R&D :** `ConvNeXt-Base` (88M paramètres).
   - **Student Edge AI :** `MobileNetV4-Small` (3.8 MB INT8) entraîné par **Distillation Hybride** (Soft-BCE + Attention Transfer).

---

## 🎯 Métriques Cibles (Sécurité Clinique SaMD)

| Métrique | Seuil Baseline | Seuil Cible (SOTA) | Justification Clinique |
| :--- | :---: | :---: | :--- |
| **Sensibilité (Recall)** | **$\ge 95.0\%$** | **$\ge 97.0\%$** | **Priorité Absolue :** Zéro lésion précancéreuse CIN2/CIN3 manquée. |
| **Spécificité** | **$\ge 80.0\%$** | **$\ge 85.0\%$** | **Efficience Système :** Éviter les actes de thermocoagulation inutiles. |
| **Score $F_2$** | **$\ge 0.88$** | **$\ge 0.93$** | Pondération du Recall $2\times$ supérieure à la Précision. |
| **AUC-ROC** | **$\ge 0.90$** | **$\ge 0.94$** | Capacité globale de discrimination binaire et tri-classe. |

---

## 📂 Structure Complète du Repository

```text
LAAFI_AI_IVA/
│
├── docs/                                 # 📚 DOCUMENTATION COMPLÈTE & SaMD CLINIQUE
│   ├── clinical/                         # Protocoles médicaux & Spécifications OMS
│   │   ├── clinical_protocol_via.md      # Protocole d'examen IVA en milieu rural (CSPS)
│   │   ├── clinical_metrics_samd.md      # Justification mathématique & éthique (Recall >= 95%)
│   │   └── risk_management_iso14971.md   # Matrice de gestion des risques ISO 14971
│   ├── architecture/                     # Conception technique & Edge AI
│   │   ├── two_stage_pipeline.md         # Stage 1 CADe (YOLO) + Stage 2 CADx (ConvNeXt)
│   │   ├── distillation_framework.md     # Knowledge Distillation Teacher -> Student
│   │   └── edge_quantization_spec.md     # Spécifications PTQ / QAT INT8 & ExecuTorch
│   ├── governance/                       # Cartes de conformité IA responsable
│   │   ├── MODEL_CARD.md                 # Fiche technique modèle (performances, limites)
│   │   └── DATA_CARD.md                  # Provenance dataset, RGPD & GroupKFold
│   └── project_management/               # Suivi de projet & Roadmaps
│       ├── ROADMAP_30_DAYS.md            # Feuille de route 30 jours (Semaines 1 à 4)
│       ├── WHOLEPROJECT_SPEC.md          # Brief scientifique et axes de Deep Search
│       └── task_trackers/                # Suivis hebdomadaires archivés
│           ├── TASK_TRACKER_WEEK2.md
│           └── TASK_TRACKER_WEEK3.md
│
├── config/                               # ⚙️ CONFIGURATION GLOBALE
│   └── config.yaml                       # Chemins, hyperparamètres et SEED = 42
│
├── src/                                  # 📦 PACKAGE MODULAIRE (Imports directs src.*)
│   ├── data/                             # Ingestion, clustering patients, augmentations, masques
│   ├── preprocessing/                    # Module Reflets-Lite (HSV)
│   ├── models/                           # Wrappers YOLO (CADe), ConvNeXt (CADx) et MobileNetV4
│   ├── losses/                           # AsymmetricFocalLoss
│   ├── distillation/                     # BinaryHybridKDLoss (Soft-BCE + Spatial Attention)
│   ├── utils/                            # Seed, métriques cliniques, calibration T_opt, Grad-CAM
│   └── train.py                          # Engine d'entraînement principal Stage 2
│
├── experiments/                          # 🧪 SCRIPTS D'EXPÉRIMENTATION DÉDIÉS
│   ├── run_ablation.py                   # Grille d'ablation FP32
│   └── run_distillation.py               # Entraînement Student par Distillation Hybride
│
├── export/                               # 📱 OPTIMISATION & DÉPLOIEMENT EDGE
│   ├── ptq_quantizer.py                  # Quantification Post-Training INT8
│   ├── qat_trainer.py                    # Fallback Quantization-Aware Training
│   ├── export_executorch.py              # Export Graphe PyTorch 2.x & ExecuTorch
│   └── evaluate_quantized.py             # Audit de perte métrique post-quantification
│
├── notebooks/                            # 📓 NOTEBOOKS COLAB & KAGGLE
│   └── LAAFI_AI_IVA_Kaggle_Master_Pipeline.ipynb # Pipeline exécutable nativement sur GPU Kaggle
│
├── scripts/                              # 🛠️ SCRIPTS AUTOMATISÉS
│   └── create_valid_notebook.py          # Générateur du notebook Master Kaggle
│
├── tests/                                # 🧪 SUITE DE TESTS (100% Locale sur CPU)
│   ├── conftest.py                       # Fixtures PyTorch & tenseurs synthétiques
│   ├── test_patient_leakage.py           # Vérification étanchéité StratifiedGroupKFold
│   ├── test_asymmetric_loss.py           # Stabilité & gradients AsymmetricFocalLoss
│   ├── test_kd_loss.py                   # Gradients & attention maps BinaryHybridKDLoss
│   ├── test_threshold_calibration.py     # Vérification contrainte Recall >= 95%
│   ├── test_cervix_transforms.py         # Test du filtre Reflets-Lite
│   └── test_models_forward.py            # Test forward passes ConvNeXt & MobileNetV4
│
├── .gitignore
├── setup.py                              # Package installation (pip install -e .)
└── README.md
```

---

## 🚀 Guide d'Utilisation

### 1. Installation en Local (Développement & Tests CPU)
```bash
git clone https://github.com/FLICKWICK226/LAAFI_AI_IVA.git
cd LAAFI_AI_IVA

# Environnement virtuel
python -m venv .venv
# Linux/macOS : source .venv/bin/activate
# Windows : .\.venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### 2. Exécution de la Suite de Tests (100% CPU en < 15 secondes)
```bash
pytest tests/ -v
```

### 3. Exécution Native sur GPU Kaggle
Le projet est conçu pour une exécution fluide et interactive sur Kaggle (Zero-Download Mode) via le notebook :
👉 `notebooks/LAAFI_AI_IVA_Kaggle_Master_Pipeline.ipynb`

---

## 📄 Licence
Projet distribué sous licence [MIT](LICENSE). Développé pour la recherche et l'assistance clinique SaMD.
