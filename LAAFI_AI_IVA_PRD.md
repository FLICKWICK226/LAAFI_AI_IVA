
# 📄 Product Requirement Document (PRD) - REVISED V2.0
## Projet : Engine IA IVA – LAAFI_AI (Dépistage du Cancer du Col de l'Utérus)

**Version :** 2.0  
**Statut :** Phase d'Ingénierie & Entraînement (Plan Révisé)  
**Datasets :** `intel-mobileodt-cervical-cancer-screening` (Éligibilité anatomique) + NIH Cervigram / VIA Subsets (Diagnostic Lésionnel)  
**Stack Technique :** Python 3.10+ | PyTorch | Albumentations | OpenVINO / ONNX

---

## 1. Description du Projet & Périmètre

Le moteur IA IVA de **LAAFI_AI** est un système de vision par ordinateur à deux étages (*CADe/CADx - Computer-Aided Detection & Diagnosis*) conçu pour assister les professionnels de santé lors de l'examen à l'acide acétique.

### Objectif Principal
Détecter la **Zone de Jonction Squamo-Columnaire (JSC)** via bounding boxes (CADe) et évaluer à la fois l'**éligibilité anatomique** (JSC visible Type 1/2 vs Type 3) et la **présence de lésions acéto-blanches précancéreuses** (VIA Positif vs VIA Négatif) (CADx) sur des clichés pris par smartphone.

---

## 2. Objectifs Cibles & Alignement Réglementaire (SaMD)

Conformément aux normes internationales de dispositifs médicaux (FDA CADe/CADx et CE MDR Classe IIb), le modèle vise un équilibre clinique optimal pour éviter la saturation des structures sanitaires secondaires.

### Métriques Clés

| Métrique | Seuil Minimal Acceptable (Baseline) | Seuil Cible (SOTA Médical) | Justification Clinique |
| :--- | :--- | :--- | :--- |
| **Sensibilité (Recall)** | **$\ge 95.0\%$** | **$97.0\%$** | **Sécurité Patient :** Réduire au strict minimum les Faux Négatifs (lésions ratées). |
| **Spécificité** | **$\ge 80.0\%$** | **$85.0\%$** | **Efficacité Système :** Éviter l'engorgement des centres de biopsie par sur-référence. |
| **AUC-ROC** | **$\ge 0.90$** | **$\ge 0.94$** | Capacité globale de discrimination (Éligibilité & Diagnostic). |
| **Score $F_2$** | **$\ge 0.88$** | **$\ge 0.92$** | Métrique de référence pondérant le Recall 2x plus que la Précision. |

---

## 3. Architecture Globale du Modèle (Pipeline 2 Étages)

1. **Étape 1 (CADe - Localisation ROI)** : Modèle **YOLOv8-Det** ou **YOLOv11-Det** (détection d'objets) entraîné sur les boîtes englobantes (`bounding_boxes.csv`) pour détecter la Zone de Jonction Squamo-Columnaire (JSC) et rogner l'image ($384 \times 384$) afin d'éliminer le spéculum, le vagin et l'arrière-plan.
2. **Étape 2 (CADx - Multi-tâche Eligibility & Diagnosis)** : Le crop de la JSC est injecté dans un backbone **ConvNeXt-Base** ou **Swin Transformer v2** pour une double prédiction :
   - **Tâche A (Éligibilité IVA)** : Classer la visibilité de la JSC (Type 1/2 = Éligible vs Type 3 = Invalide/Endocervical) via le dataset Intel-MobileODT.
   - **Tâche B (Diagnostic Lésionnel)** : Classer la présence de lésion (VIA Positif vs Négatif) via les sous-ensembles cliniquement annotés (NIH Cervigram / VIA Subsets).

---

## 4. Architecture de Simulation des Bruits de Terrain

Pour garantir la robustesse sur des appareils dégradés ou bas de gamme, le pipeline d'entraînement intègre un module de génération de bruits physiques et biologiques optimisé :

- **Sang & Glaire (Générés Hors-Ligne)** : Masques de Perlin pré-calculés stockés dans `./data/synthetic_masks/` pour éviter les ralentissements CPU pendant l'entraînement. Injection dynamique de couleur spectrale (Sang : R=140-180, G=10-30, B=10-30 / Glaire : R=220, G=210, B=180) fusionnée avec un facteur Alpha $\alpha \in [0.15, 0.45]$.
- **Reflets Spéculaires (Flash/LED)** : Seuil de luminance élevé ($V > 240$ sur HSV) avec dilatation morphologique pour simuler la lumière réfléchie sur le liquide acétique.
- **Flous Optiques** : Disk/Defocus Kernel (défocalisation du smartphone) et Linear Motion Kernel (tremblement de l'opérateur).

---

## 5. Module Python : Pipeline d'Augmentation Optimisé

```python
# src/data/augmentations.py

import os
import cv2
import numpy as np
import albumentations as A

class FastPerlinNoiseLoader:
    """Chargeur rapide de masques de bruit de Perlin pré-générés pour éviter les boucles CPU."""
    
    def __init__(self, masks_dir="./data/synthetic_masks"):
        self.masks_dir = masks_dir
        self.mask_files = []
        if os.path.exists(masks_dir):
            self.mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.npy')]

    def add_blood_or_mucus(self, image, noise_type='blood', max_alpha=0.4):
        if not self.mask_files:
            return image # Fallback si les masques ne sont pas encore générés
            
        h, w, c = image.shape
        mask_path = np.random.choice(self.mask_files)
        perlin = np.load(mask_path)
        
        # Redimensionner le masque si nécessaire
        if perlin.shape != (h, w):
            perlin = cv2.resize(perlin, (w, h), interpolation=cv2.INTER_LINEAR)
            
        mask = (perlin > 0.6).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        overlay = image.copy()
        if noise_type == 'blood':
            overlay[:, :, 0] = np.random.randint(10, 30)   # B
            overlay[:, :, 1] = np.random.randint(10, 30)   # G
            overlay[:, :, 2] = np.random.randint(130, 180) # R
        elif noise_type == 'mucus':
            overlay[:, :, 0] = np.random.randint(160, 190) # B
            overlay[:, :, 1] = np.random.randint(200, 220) # G
            overlay[:, :, 2] = np.random.randint(210, 230) # R

        alpha = mask[:, :, np.newaxis] * max_alpha
        blended = (image * (1 - alpha) + overlay * alpha).astype(np.uint8)
        return blended

def build_iva_augmentation_pipeline():
    """Pipeline d'augmentation robuste pour imagerie de terrain (Hue bridé pour l'IVA)."""
    return A.Compose([
        A.OneOf([
            A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.4), p=0.5),
            A.MotionBlur(blur_limit=(3, 9), p=0.5),
        ], p=0.5),
        A.RandomSunFlare(
            flare_roi=(0.2, 0.2, 0.8, 0.8), 
            angle_lower=0, angle_upper=1, 
            src_radius=80, p=0.3
        ),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05, p=0.5), # Hue bridé à 0.05
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.5),
        ], p=0.4),
        A.Vignette(vignette_fade_limit=0.25, p=0.3),
        A.ChromaticAberration(primary_distortion_limit=0.04, p=0.3),
    ])
```

## 6. Architecture Modulaire des Dossiers & Workflow Drive

```text
LAAFI_AI_IVA/
│
├── config/
│   └── config.yaml               # Hyperparamètres, chemins relatifs, seuil T, SEED = 42
│
├── notebooks/                    # 📓 NOTEBOOKS COLAB EXÉCUTABLES (PERSISTANTS SUR DRIVE)
│   ├── 01_setup.ipynb            # Montage Drive, SEED=42, Userdata secrets, arborescence outputs
│   ├── 02_data_preparation.ipynb # Ingestion, Clustering patients, Splits & Perlin Offline
│   ├── 03_train.ipynb            # Stage 1 (YOLO-Det JSC) + Stage 2 (Swin/ConvNeXt) + Fine-tuning
│   └── 04_eval_inference.ipynb   # Plotting figures, calibration seuil T & Export ONNX
│
├── data/                         # 💾 STOCKAGE PERSISTANT SUR DRIVE (ZÉRO CACHE VOLATILE)
│   ├── raw/                      # Dataset brut Intel-MobileODT + NIH Cervigram / VIA Subsets
│   ├── synthetic_masks/          # 1000 masques Perlin .npy pré-générés hors-ligne
│   └── processed/                # Crops JSC (384x384) et splits (train/val/test.csv)
│
├── models/                       # 💾 CHECKPOINTS PERSISTANTS SUR DRIVE
│   ├── checkpoints/              # Checkpoints intermédiaires (.pt) & best_model.pt
│   └── exported/                 # Modèle final quantifié (ONNX / OpenVINO)
│
├── outputs/                      # 📊 DOSSIER DES RÉSULTATS, VISUELS & RAPPORTS
│   ├── figures/                  # confusion_matrix.png, roc_curve.png, gradcam_audit/
│   ├── reports/                  # metrics_report.csv, classification_report.json
│   └── logs/                     # Logs d'entraînement TensorBoard / CSV
│
├── src/                          # 📦 PACKAGE PYTHON MODULAIRE (DÉVELOPPÉ DANS L'IDE)
│   ├── __init__.py
│   ├── data/
│   │   ├── downloader.py         # Ingestion idempotent (Kaggle -> Drive)
│   │   ├── cluster_patients.py   # Regroupement des patients par similarité / DBSCAN
│   │   ├── dataset.py            # PyTorch Dataset Class
│   │   └── augmentations.py      # Chargement masques Perlin + Albumentations
│   │
│   ├── models/
│   │   ├── detector_jsc.py       # Wrapper YOLOv8/v11-Det (Stage 1)
│   │   └── classifier_lesion.py  # Wrapper Swin-v2 / ConvNeXt (Stage 2 multi-tâche)
│   │
│   ├── utils/
│   │   ├── seed.py               # Fonction seed_everything(42)
│   │   ├── metrics.py            # Focal Loss, Sensibilité, Spécificité, F2
│   │   └── visualization.py      # Heatmaps Grad-CAM et Matrice de Confusion
│   │
│   └── train.py                  # Engine de training principal
│
├── requirements.txt
└── setup.py                      # Package installation (pip install -e .)
```

## 7. Script de Téléchargement Idempotent via Clé API Colab

Python

```python
# src/data/downloader.py

import os
import sys

def download_intel_mobileodt_dataset(target_data_path="./data/raw"):
    """Télécharge le dataset Kaggle 'intel-mobileodt-cervical-cancer-screening' via KAGGLE_API_KEY."""
    os.makedirs(target_data_path, exist_ok=True)
    
    train_dir = os.path.join(target_data_path, "train")
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        print(f"✅ Dataset déjà présent dans : {target_data_path}")
        return

    print("📥 Initialisation de l'accès aux données via la clé API Colab (KAGGLE_API_KEY)...")
    if "google.colab" in sys.modules:
        from google.colab import userdata
        api_key = userdata.get("KAGGLE_API_KEY") or userdata.get("KAGGLE_TOKEN")
        if api_key:
            os.environ["KAGGLE_API_KEY"] = api_key
            os.environ["KAGGLE_KEY"] = api_key

    import kaggle
    dataset_name = "intel-mobileodt-cervical-cancer-screening"
    kaggle.api.dataset_download_files(dataset_name, path=target_data_path, unzip=True)
    print(f"🎉 Téléchargement et extraction terminés dans : {target_data_path}")

if __name__ == "__main__":
    download_intel_mobileodt_dataset()
```

## 8. Protocole de Validation Clinique

1. **Splitting par Patient** : Regroupement strict des images par patiente via le script `cluster_patients.py` (en utilisant des caractéristiques visuelles ResNet + DBSCAN ou la cartographie communautaire des doublons) pour éliminer tout risque de _Data Leakage_ avant d'effectuer un `GroupKFold`.
    
2. **Évaluation Multi-critère** : Validation séparée de l'Éligibilité (Type 1/2 vs 3) et du Diagnostic (Positif vs Négatif) pour assurer que la décision clinique est sécurisée.

3. **Matrice de Confusion Sécurisée** : Les Faux Négatifs (lésions ratées) sont pénalisés de 3x à 5x dans la fonction de perte (Focal Loss, $\gamma = 2.0$).
    
4. **Audit visuel Grad-CAM** : Génération systématique des cartes d'attention visuelle sur les crops de validation pour vérifier que le classifieur focalise son attention sur les signes d'acéto-blanchiment de l'épithélium et non sur les bruits (sang, glaire, spéculum).
    
