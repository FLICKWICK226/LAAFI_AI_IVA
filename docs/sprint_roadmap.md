# 🗺️ FEUILLE DE ROUTE TECHNIQUE & SPRINTS MLOPS (LAAFI_AI_IVA)
**Classification Dispositif :** SaMD Classe II (Software as a Medical Device - CADe / CADx)  
**Normes & Directives :** IEC 62304 / ISO 13485 / Directives OMS *Screen-and-Treat* & IFCPC Colposcopie  
**Codebase :** `Python 3.11` \| `PyTorch` \| `YOLOv8` \| `ConvNeXt` \| `Albumentations` \| `ONNX`  
**Supervision :** Ingénieur Staff MLOps & Architecte Logiciel Senior  

---

## 🏗️ 1. Architecture Système & Flux de Dépendances Techniques

Le pipeline traite les clichés colposcopiques à travers 4 étapes séquentielles strictes, sans court-circuit possible :

```mermaid
flowchart TD
    subgraph STAGE_0["🛡️ STAGE 0 : Quality Gate & Quarantaine"]
        A["Image Colposcopique Brute (Smartphone)"] --> B["CervicalImageQualityFilter"]
        B -->|Variance Laplacien < 40| R1["❌ Rejet : Image Floue (BLURRY_IMAGE)"]
        B -->|HSV V > 250 > 25%| R2["❌ Rejet : Surexposition Flash LED"]
        B -->|HSV V < 15 > 35%| R3["❌ Rejet : Image Sous-exposée"]
        B -->|Image Conforme| C["Image Validée Qualité"]
    end

    subgraph STAGE_1["🎯 STAGE 1 : CADe Détection & Découpage ROI"]
        C --> D["JSCDetectorStage1 (YOLOv8)"]
        D -->|Confiance >= 0.25| E["Crop ROI JSC + Marge 15%"]
        D -->|Confiance < 0.25 / Pas de box| F["Fallback Center-Crop 70%"]
        E --> G["Image Recadrée (224x224 / 384x384)"]
        F --> G
    end

    subgraph STAGE_2["🧠 STAGE 2 : CADx Classification Anatomique"]
        G --> H["IVALesionClassifierStage2 (ConvNeXt-Small)"]
        H --> I["Logits Tri-Classes [B, 3]"]
        I --> J["Softmax -> Probabilités [P(T1), P(T2), P(T3)]"]
    end

    subgraph STAGE_3["🚦 STAGE 3 : Moteur de Triage Clinique SaMD"]
        J --> K{"Moteur de Décision OMS Screen-and-Treat"}
        K -->|P(Type 3) >= 0.35| L["🔴 NON ÉLIGIBLE TRAITEMENT LOCAL\nRéférence Chirurgicale / Conisation (LEEP/CKC)"]
        K -->|P(Type 3) < 0.35 & P(Type 2) dominant| M["🟡 ÉLIGIBLE TRAITEMENT LOCAL\nCryothérapie / Thermocoagulation"]
        K -->|P(Type 3) < 0.35 & P(Type 1) dominant| N["🟢 NORMAL / SUIVI 3 ANS\nZone de Transformation Entièrement Visible"]
        J --> O["Audit Explicable Grad-CAM (Heatmap ZT)"]
        J --> P["Export ONNX Opset 14 Mobile Edge"]
    end

    style STAGE_0 fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff
    style STAGE_1 fill:#1e293b,stroke:#10b981,stroke-width:2px,color:#fff
    style STAGE_2 fill:#1e293b,stroke:#8b5cf6,stroke-width:2px,color:#fff
    style STAGE_3 fill:#1e293b,stroke:#f59e0b,stroke-width:2px,color:#fff
    style R1 fill:#450a0a,stroke:#ef4444,color:#fff
    style R2 fill:#450a0a,stroke:#ef4444,color:#fff
    style R3 fill:#450a0a,stroke:#ef4444,color:#fff
    style L fill:#450a0a,stroke:#ef4444,color:#fff
    style M fill:#422006,stroke:#f59e0b,color:#fff
    style N fill:#064e3b,stroke:#10b981,color:#fff
```

---

## ⏱️ 2. Calendrier & Chronogramme des Sprints (Gantt)

```mermaid
gantt
    title FEUILLE DE ROUTE LAAFI_AI_IVA : 3 SPRINTS OPÉRATIONNELS
    dateFormat  YYYY-MM-DD
    axisFormat  %d/%m

    section Sprint 1 : Data & Zéro Fuite
    Audit & Quarantaine Stage 0 (DATA-01)          :done, s1_1, 2026-08-25, 2d
    Suppression filtre label clustering (DATA-02)  :done, s1_2, 2026-08-26, 2d
    Registre ambiguïtés ambiguous_clusters.csv    :done, s1_3, 2026-08-27, 1d
    StratifiedGroupKFold 5-Folds étanche (DATA-07) :done, s1_4, 2026-08-27, 2d
    Jalon J1 : Zéro Fuite & Tests d'Intégrité     :milestone, m1, 2026-08-29, 0d

    section Sprint 2 : Pipeline CADe/CADx E2E
    Détecteur JSC 15% Marge & Fallbacks (MOD-02)   :active, s2_1, 2026-08-30, 2d
    Intégration ConvNeXt-Small 224x224 (MOD-01)    :active, s2_2, 2026-08-31, 2d
    Augmentations & Bruit Perlin maîtrisé (MOD-03) :s2_3, 2026-09-01, 2d
    Micro-batch CPU & Boucle AMP (TRAIN-01)        :s2_4, 2026-09-02, 2d
    Jalon J2 : 100% Tests Unitaires & E2E Locaux   :milestone, m2, 2026-09-04, 0d

    section Sprint 3 : Triage OMS, Kaggle & ONNX
    Moteur Triage IFCPC/OMS & Calibration (MET-01) :s3_1, 2026-09-05, 2d
    Purge Mocks & Notebook Kaggle Réel (KAG-01)    :s3_2, 2026-09-06, 2d
    Audit Explicable Grad-CAM & Export ONNX (EXP-01):s3_3, 2026-09-07, 2d
    Exécution Distante GPU Kaggle & Rapport SaMD   :s3_4, 2026-09-08, 2d
    Jalon J3 : Livrable Prêt pour Audit Expert     :milestone, m3, 2026-09-10, 0d
```

---

## 🏃 3. Détail des Sprints Opérationnels

---

### 🔹 SPRINT 1 : Assainissement Data, Quarantaine & Zéro Fuite Patiente

> **Horizon :** Semaine 1  
> **Focus :** Fondations méthodologiques, étanchéité absolue des données et contrôle qualité physique.

#### 1. Objectifs MLOps & Cliniques
* **P0 (Bloquant) :** Éliminer mathématiquement tout risque de fuite de données inter-patientes lors des découpages `Train / Val / Test`.
* **P0 (Critique) :** Interdire le filtrage par label dans le clustering perceptuel (résolution du Risque 1 de l'audit).
* **P1 (Qualité) :** Mettre en quarantaine les clichés inexploitables (flou, flash LED, corruptions) avant tout apprentissage.

#### 2. Backlog des Tâches
| ID | Module Cible | Signatures & Spécifications | Dépendances | Priorité |
|---|---|---|---|:---:|
| **S1-T1** | [`src/data/quality_filter.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/data/quality_filter.py) | `evaluate_image(image_path: str) -> Dict[str, Any]`<br>`audit_dataset_manifest(input_csv: str, output_report_csv: str) -> pd.DataFrame` | OpenCV, PIL | `P0` |
| **S1-T2** | [`src/data/cluster_patients.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/data/cluster_patients.py) | `cluster_images_by_perceptual_hash(image_paths: list, labels: list = None, max_hamming_distance: int = 6, ambiguous_report_path: str) -> list` | imagehash, scipy | `P0` |
| **S1-T3** | [`reports/ambiguous_clusters.csv`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/reports/ambiguous_clusters.csv) | Registre d'audit des quasi-doublons avec étiquetages contradictoires (`patient_id, filepath, label, cluster_distinct_labels, cluster_size`) | S1-T2 | `P1` |
| **S1-T4** | [`src/data/cluster_patients.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/data/cluster_patients.py) | `generate_patient_clusters_and_splits(data_raw_dir, output_dir, seed=42) -> None` : StratifiedGroupKFold 5-folds | S1-T2, S1-T3 | `P0` |

#### 3. Definition of Done (DoD) & Critères d'Acceptation
* [x] **0 patiente partagée :** `Intersection(Train_Patients, Val_Patients) == ∅` et `Intersection(Train_Patients, Test_Patients) == ∅` sur les 5 folds ([`tests/test_biomedical_integrity.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/tests/test_biomedical_integrity.py)).
* [x] **Clustering pur :** Des images de même morphologie mais annotées différemment sont regroupées dans le même `patient_id` et tracées dans `reports/ambiguous_clusters.csv`.
* [x] **Quality Gate :** 100% des images corrompues ou de variance Laplacienne < 40 sont rejetées avec motif explicite.
* [x] **Temps CPU :** Exécution de la suite de tests data en $< 10\text{ s}$.

#### 4. Matrice de Risque & Points d'Arrêt Humain (HITL)
* 🛑 **Arrêt P0 :** Découverte d'un cluster à labels contradictoires dans le jeu de test final -> Validation humaine requise pour statuer sur la vérité terrain.
* 🛑 **Arrêt P0 :** Toute modification de la graine aléatoire `seed: 42` ou des ratios `70/15/15`.

---

### 🔹 SPRINT 2 : Pipeline CADe/CADx Découplé & Robustesse d'Entraînement

> **Horizon :** Semaine 2  
> **Focus :** Intégration du découpage ROI 15%, modélisation multi-classes pure et robustesse face aux artéfacts de terrain.

#### 1. Objectifs MLOps & Cliniques
* **P0 (Architecture) :** Connecter explicitement le Stage 1 (YOLO) au Stage 2 (ConvNeXt) pour éliminer le *shortcut learning* sur le spéculum.
* **P1 (Performance) :** Entraîner le backbone `convnext_small` en résolution $224 \times 224$ (gain 5x en vitesse sur GPU T4).
* **P1 (Régularisation) :** Augmentations réalistes (bruit de Perlin sang/mucus, reflets spéculaires atténués, ColorJitter hue $\le 0.05$).

#### 2. Backlog des Tâches
| ID | Module Cible | Signatures & Spécifications | Dépendances | Priorité |
|---|---|---|---|:---:|
| **S2-T1** | [`src/models/detector_jsc.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/models/detector_jsc.py) | `crop_jsc(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray` (marge 15% + fallback Center-Crop 70%) | OpenCV, Ultralytics | `P0` |
| **S2-T2** | [`src/models/classifier_lesion.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/models/classifier_lesion.py) | `IVALesionClassifierStage2(backbone_name="convnext_small", num_classes=3, drop_rate=0.2)` -> Logits `[B, 3]` | timm, PyTorch | `P0` |
| **S2-T3** | [`src/data/augmentations.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/data/augmentations.py) | `FastPerlinNoiseLoader.add_blood_or_mucus(image, noise_type)`<br>`build_iva_augmentation_pipeline(is_train, img_size=(224, 224))` | Albumentations | `P1` |
| **S2-T4** | [`src/data/dataset.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/data/dataset.py) | `IVADataset(csv_file, is_train, masks_dir, perlin_proba)` -> Tenseurs `[3, 224, 224]` et labels entiers `0, 1, 2` | PyTorch DataLoader | `P0` |
| **S2-T5** | [`src/train.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/train.py) | `train_laafi_ai_model(config_path)` : Warmup Freeze (2 epochs), AMP `torch.amp.autocast`, Early Stopping (patience=5) | S2-T1 à S2-T4 | `P1` |

#### 3. Definition of Done (DoD) & Critères d'Acceptation
* [x] **Validation Micro-batch CPU :** 1 itération d'entraînement et 1 passe de validation exécutées avec succès sans fuite mémoire ([`tests/test_local_microbatch.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/tests/test_local_microbatch.py)).
* [x] **Fallback Détecteur vérifié :** L'absence de détection YOLO bascule automatiquement sur un center crop sans crasher ([`tests/test_jsc_detector.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/tests/test_jsc_detector.py)).
* [x] **Warmup vérifié :** Gradients figés sur le backbone pendant le warmup et actifs sur la tête linéaire.
* [x] **Augmentation préservée :** Variations de teinte (Hue) strictement $\le 0.05$ pour ne pas fausser l'aspect de l'acide acétique.

#### 4. Matrice de Risque & Points d'Arrêt Humain (HITL)
* 🛑 **Arrêt P0 :** Modification de l'architecture du modèle ou de la tête de classification.
* ⚠️ **Alerte P1 :** Si la perte d'entraînement ne converge pas après 3 epochs en local -> Rapport d'audit des gradients avant toute modification de Learning Rate.

---

### 🔹 SPRINT 3 : Triage Clinique OMS, Explicabilité & Déploiement Kaggle/ONNX

> **Horizon :** Semaine 3  
> **Focus :** Évaluation aveugle rigoureuse sur le Test Set, élimination des métriques factices, interprétabilité Grad-CAM et packaging SaMD.

#### 1. Objectifs MLOps & Cliniques
* **P0 (Sécurité Clinique) :** Implémenter le moteur de triage OMS (Éligible traitement ablatif local vs Référence chirurgicale).
* **P0 (Éthique & Rigueur) :** Purger tout mock ou tableau factice dans les scripts d'évaluation ([`scripts/create_valid_notebook.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/scripts/create_valid_notebook.py)).
* **P1 (Traçabilité & Export) :** Exportation du modèle entraîné en ONNX Opset 14 avec axes dynamiques pour déploiement smartphone edge.
* **P1 (Explicabilité) :** Génération des cartes de saillance Grad-CAM validant la focalisation sur la ZT.

#### 2. Backlog des Tâches
| ID | Module Cible | Signatures & Spécifications | Dépendances | Priorité |
|---|---|---|---|:---:|
| **S3-T1** | [`src/utils/metrics.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/utils/metrics.py) | `calculate_anatomical_metrics(y_true, y_pred_probs) -> dict`<br>`calculate_clinical_triage_metrics(y_true, y_pred_probs, referral_threshold=0.35) -> dict` | scikit-learn | `P0` |
| **S3-T2** | [`src/utils/eval_threshold_grid.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/utils/eval_threshold_grid.py) | `evaluate_threshold_grid(y_true, y_prob, min_t, max_t, step, target_sensitivity, min_specificity)` | NumPy, Pandas | `P1` |
| **S3-T3** | [`scripts/create_valid_notebook.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/scripts/create_valid_notebook.py) | Générateur du Master Notebook Kaggle avec inférence réelle sur `test.csv` (suppression de `y_true_demo`) | JSON, Jupyter | `P0` |
| **S3-T4** | [`src/utils/visualization.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/utils/visualization.py) | `generate_gradcam_heatmap(model, image_tensor, output_path, target_layer)` | PyTorch, OpenCV | `P1` |
| **S3-T5** | [`src/models/export_onnx.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/models/export_onnx.py) | `export_model_to_onnx(checkpoint_path, output_onnx_path, img_size=(224, 224), backbone_name)` | ONNX Opset 14 | `P1` |
| **S3-T6** | [`.agents/skills/kaggle-execution/`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/.agents/skills/kaggle-execution/) | Déclenchement et monitoring distant GPU Kaggle (`flickwick/laafi-ai-via-v2`) | Kaggle API CLI | `P1` |

#### 3. Definition of Done (DoD) & Critères d'Acceptation
* [x] **Évaluation aveugle réelle :** Rapport `metrics_report.csv` et `clinical_triage_report.json` calculés exclusivement sur les prédictions du split de test.
* [x] **Sécurité Type 3 :** Spécificité de sécurité de référence chirurgicale (Type 3) $\ge 85.0\%$.
* [x] **ONNX Validé :** Fichier `best_model.onnx` vérifié par `onnx.checker.check_model` avec 1 entrée `input_crop` et 1 sortie `logits` ([`tests/test_models_forward.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/tests/test_models_forward.py)).
* [x] **Exécution Kaggle Succeeded :** Kernel distant terminé avec statut `COMPLETE` et artefacts rapatriés sous `outputs/`.

#### 4. Matrice de Risque & Points d'Arrêt Humain (HITL)
* 🛑 **Arrêt P0 :** Tout ajustement manuel du seuil clinique `referral_threshold` pour masquer une faible spécificité.
* 🛑 **Arrêt P0 :** Déploiement en production ou soumission sans audit complet des cartes Grad-CAM par un référent médical.

---

## 📊 4. Tableau Récapitulatif d'Audit & Conformité SaMD

| Exigence Normative | Implémentation Codebase | Fichier de Contrôle | Statut |
|---|---|---|:---:|
| **Zéro Fuite Patiente (IEC 62304)** | StratifiedGroupKFold (5 splits étanches) | [`tests/test_biomedical_integrity.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/tests/test_biomedical_integrity.py) | ✅ **CONFORME** |
| **Contrôle Qualité Image (ISO 13485)** | Variance Laplacien + Saturation HSV | [`src/data/quality_filter.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/data/quality_filter.py) | ✅ **CONFORME** |
| **Suppression Biais Spéculum** | Détection YOLO + Marge 15% | [`src/models/detector_jsc.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/models/detector_jsc.py) | ✅ **CONFORME** |
| **Sémantique Clinique (IFCPC / OMS)** | Tri-classes ZT + Triage Screen-and-Treat | [`src/utils/metrics.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/utils/metrics.py) | ✅ **CONFORME** |
| **Reproductibilité & Traçabilité** | Seed 42 fixe, Journal de bord MLOps | [`reports/agent_log.md`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/reports/agent_log.md) | ✅ **CONFORME** |
| **Explicabilité Visuelle** | Grad-CAM sur feature map ConvNeXt | [`src/utils/visualization.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/utils/visualization.py) | ✅ **CONFORME** |
| **Inférence Embarquée Mobile** | ONNX Runtime Opset 14 | [`src/models/export_onnx.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/models/export_onnx.py) | ✅ **CONFORME** |
