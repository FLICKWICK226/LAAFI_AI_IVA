# 🔬 RAPPORT D'AUDIT TECHNIQUE ET MÉTHODOLOGIQUE DU CODEBASE
**Date :** 2026-08-25  
**Projet :** LAAFI_AI_IVA (SaMD Classe II CADe / CADx)  
**Auteur :** Ingénieur Staff MLOps & Architecte Logiciel Senior  

---

## 1. Recensement des Modules & Architecture

| Module | Rôle dans le Pipeline | Fichiers Clés |
|---|---|---|
| **Contrôle Qualité (Stage 0)** | Filtrage flou Laplacien, exposition LED, corruption | `src/data/quality_filter.py` |
| **Données & Clustering** | Indexation, Perceptual Hash (dHash/aHash), Splits GroupKFold | `src/data/cluster_patients.py`, `src/data/dataset.py`, `src/data/augmentations.py` |
| **Détecteur CADe (Stage 1)** | Détection YOLO de la JSC + découpage ROI 15% marge | `src/models/detector_jsc.py` |
| **Classifieur CADx (Stage 2)** | ConvNeXt-Small 3-classes unifié (Types 1/2/3) | `src/models/classifier_lesion.py`, `src/models/student_model.py` |
| **Entraînement & Optim** | Boucle AMP, Warmup Freeze, Early Stopping, Checkpoints | `src/train.py`, `config/config.yaml` |
| **Métriques & Triage** | Calibration de seuil T, métriques cliniques & anatomiques OMS | `src/utils/metrics.py`, `src/utils/eval_threshold_grid.py` |
| **Export Mobile** | Exportateur ONNX avec axes dynamiques | `src/models/export_onnx.py` |

---

## 2. Audit des Vulnérabilités & Risques Méthodologiques (Phase 0)

### 🚨 Risque 1 : Fuite de Données par Filtrage de Label dans le Clustering Patient
- **Fichier / Ligne :** [`src/data/cluster_patients.py:166`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/data/cluster_patients.py#L166)
- **Code incriminé :**
  ```python
  if labels[idx_i] == labels[idx_j]:
      row_ind.append(idx_i)
      col_ind.append(idx_j)
  ```
- **Diagnostic sans concession :** Ce filtre est une **aberration méthodologique**. Si deux clichés quasi-identiques de la même patiente portent des annotations contradictoires (bruit d'étiquetage ou variation d'expert), le filtre ignore leur adjacence. Le graphe de connexité les sépare alors en deux `patient_id` distincts, permettant à une image de se retrouver en `train` et à son quasi-clone en `val` ou `test`.
- **Action corrective exigée :** 
  1. Supprimer le filtre `if labels[idx_i] == labels[idx_j]`.
  2. Regrouper par pure distance de Hamming (<= 6).
  3. Détecter les clusters contenant des labels hétérogènes et les isoler dans [`reports/ambiguous_clusters.csv`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/reports/ambiguous_clusters.csv) pour revue d'experts sans supprimer aveuglément les données.

---

### 🚨 Risque 2 : Court-Circuit Silencieux du Détecteur YOLO (Stage 1)
- **Fichier / Ligne :** [`src/data/dataset.py:78-83`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/data/dataset.py#L78-L83)
- **Code incriminé :**
  ```python
  if os.path.exists(img_path):
      image = cv2.imread(img_path)
  # Aucune invocation de JSCDetectorStage1.crop_jsc(image)
  ```
- **Diagnostic :** Alors que `src/models/detector_jsc.py` implémente un découpage de la JSC avec 15% de marge, `IVADataset` charge l'image complète brute. Le classifieur Stage 2 s'entraîne donc sur l'image globale incluant le spéculum et les parois vaginales (Shortcut Learning), sauf si des images pré-découpées sont fournies en amont.
- **Action corrective :** Permettre à `IVADataset` d'intégrer optionnellement le découpage ou documenter explicitement le mode image-entière vs ROI-crop dans le protocole de test bout-en-bout.

---

### ⚠️ Risque 3 : Binarisation Indue et Assimilation Type 1/2/3 ↔ Cancer
- **Fichier / Ligne :** [`src/train.py:333-334`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/train.py#L333-L334)
- **Code incriminé :**
  ```python
  prob_positive = (probs[:, 1] + probs[:, 2]).cpu().numpy()
  val_targets.extend((targets > 0).cpu().numpy())
  ```
- **Diagnostic :** La classification IFCPC (Colposcopie Internationale) définit :
  - **Type 1** : ZT ectocervicale, entièrement visualisable.
  - **Type 2** : ZT avec composante endocervicale, entièrement visualisable.
  - **Type 3** : ZT avec composante endocervicale non visualisable (examen incomplet / non éligible au traitement ablatif ambulatoire).
  
  Assimiler `(targets > 0)` à une détection de lésion maligne est une **erreur de sémantique clinique**. En pratique SaMD / OMS :
  - `Type 1 + Type 2` = Éligible au traitement ablatif local (cryothérapie / thermocoagulation).
  - `Type 3` = Inéligible au traitement local -> Référer en chirurgie / conisation.
- **Action corrective :** Les métriques multi-classes doivent être évaluées via `calculate_anatomical_metrics` (Accuracy, Macro-F1 par classe) et le triage via `calculate_clinical_triage_metrics`.

---

### 🚨 Risque 4 : Données de Validation Factices / Hardcodées dans le Notebook Kaggle
- **Fichier / Ligne :** [`scripts/create_valid_notebook.py:208-245`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/scripts/create_valid_notebook.py#L208-L245)
- **Code incriminé :**
  ```python
  y_true_demo = np.array([0]*60 + [1]*40)
  y_pred_demo = np.array([0]*56 + [1]*4 + [0]*2 + [1]*38)
  metrics_summary = pd.DataFrame([{"metric_name": "Sensibilité (Recall)", "target_clinical": ">= 95.0%", "value": "95.0%"}, ...])
  ```
- **Diagnostic :** La cellule 6 du notebook de génération injecte un faux jeu de prédictions `y_true_demo` et `y_pred_demo` pour dessiner des courbes ROC et matrices de confusion artificiellement parfaites. Dans un dispositif médical SaMD, c'est une fraude méthodologique inacceptable.
- **Action corrective :** Remplacer le bloc par l'inférence réelle sur le `test.csv` avec le checkpoint `best_model.pt`.

---

## 3. Plan d'Action & Statut des Phases

| Phase | Description | Statut |
|---|---|---|
| **Phase 0** | Audit approfondi du Codebase et identification des failles | ✅ **TERMINÉ** |
| **Phase 1** | Tests unitaires isolés (CPU < 30s) pour tous les composants | ⏳ En cours d'extension |
| **Phase 2** | Tests d'intégrité biomédicale (Zéro fuite, Clustering pur, Labels) | ⏳ En cours |
| **Phase 3** | Tests d'intégration bout-en-bout (Stage 0 -> Stage 1 -> Stage 2) | ⏳ En cours |
| **Phase 4** | Banc distant GPU Kaggle (Dry-run, Mémoire OOM, Non-régression) | ⏳ Prévu après validation locale |
