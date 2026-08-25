# 🏥 RAPPORT D'AUDIT TECHNIQUE, MÉTHODOLOGIQUE & CLINIQUE (v2.0)
**Projet :** `LAAFI_AI_IVA` — Système CADe/CADx pour le dépistage du cancer du col de l'utérus par VIA (Afrique subsaharienne / Burkina Faso)  
**Rôle de l'auditeur :** Expert Senior en Vision Médicale, Ingénierie ML & SaMD  
**Date :** 25 Août 2026  
**Fichier de référence :** `config/config.yaml`  

---

## ⚡ Résumé Exécutif

> **Note globale :** **3.5 / 10** — **VERDICT : NON-DÉPLOYABLE EN ÉTAT (RISQUE CLINIQUE CRITIQUE)**  
> Le projet `LAAFI_AI_IVA` présente une structure de code modulaire et une intention louable de déploiement en milieu contraint (CMA/CSPS). Cependant, il repose sur une **erreur biomédicale majeure** : l'assimilation des types anatomiques de zone de transformation du dataset Intel-MobileODT (Type 1/2/3) à des stades pathologiques de cancer (Normal/CIN/SIL/Dysplasie). À cela s'ajoutent une fuite de données résiduelle dans le clustering `imagehash`, un CADe YOLO court-circuité à l'entraînement, et un seuillage forcé masquant une spécificité effondrée. En l'état actuel, ce système ne peut en aucun cas être testé ni déployé en environnement clinique.

---

## 🔬 Audit Détaillé par Axe

---

### AXE 1 : Objectif & Pertinence Clinique

#### (a) Constat
- Le projet établit une confusion sémantique et clinique directe :
  - Dans la description : `Type 1 = CIN`, `Type 2 = SIL`, `Type 3 = Dysplasie`.
  - Dans `src/models/classifier_lesion.py` : `Type 1 (Normal / Dépistage Négatif)`, `Type 2 (Lésionnel / Dépistage Positif)`, `Type 3 (Non Visualisable / Incomplet)`.
  - Dans `src/train.py` (l. 334) : la décision binaire de positivité est calculée aveuglément par `(targets > 0)` et `prob_positive = probs[:, 1] + probs[:, 2]`.
- **Vérité clinique & anatomique (IFCPC 2011 / OMS) :**  
  Dans le dataset *Intel & MobileODT Cervical Cancer Screening* (Kaggle 2017), les labels `Type_1`, `Type_2`, `Type_3` décrivent **exclusivement l'anatomie de la Zone de Transformation (ZT) / Jonction Squamo-Columnaire (JSC)** :
  - **Type 1 :** ZT entièrement ectocervicale, 100% visible.
  - **Type 2 :** ZT avec composante endocervicale mais 100% visible (les berges sont visualisables).
  - **Type 3 :** ZT s'étendant dans l'endocol, non entièrement visualisable (anatomie masquée).

#### (b) Risque
- **Risque Mortel de Faux Négatifs :** Une patiente avec un col de morphologie `Type 1` peut présenter un cancer invasif ou un CIN3 de haut grade. En binarisant `Type 1 == 0 (Négatif)`, l'algorithme renverra chez elle une femme atteinte d'un cancer avancé en lui certifiant qu'elle est saine.
- **Surmédicalisation massive de Faux Positifs :** Une femme saine ménopausée avec une ZT de `Type 3` (physiologique après 50 ans) sera étiquetée "Positive / Cancéreuse", déclenchant des examens invasifs et une angoisse injustifiée.
- **Nullité réglementaire SaMD :** Présenter ce modèle aux autorités de santé (Ministère de la Santé du Burkina Faso, OMS) comme un "détecteur de cancer" est une faute méthodologique rédhibitoire.

#### (c) Recommandations Priorisées
1. **[P0 - Immédiat]** **Recadrer l'usage du dataset Intel-MobileODT :** Ce dataset sert uniquement à prédire **l'éligibilité anatomique au traitement ablatif** (une ZT Type 3 est inéligible à la thermocoagulation/cryothérapie en CSPS et nécessite une conisation/LEEP en centre de référence), **ET NON à diagnostiquer le cancer**.
2. **[P0 - Immédiat]** **Bannir la binarisation `(targets > 0)` :** Évaluer ce dataset en classification anatomique pure à 3 classes (Macro-F1, Macro-AUC, Matrice de confusion 3x3).
3. **[P1]** **Acquisition de données annotées par histopathologie (Gold Standard) :** Pour un véritable dépistage du cancer IVA, intégrer des datasets avec confirmation biopsique (CIN2+/CIN3/Cancer) (ex. cohortes NCI/NIH ou partenariat local CHU Yalgado / Bogodogo).

---

### AXE 2 : Données & Préparation

#### (a) Constat
- Le dataset brut compte ~8 215 images (train ~1 481, additional ~6 734, test non annoté), caractérisées par un fort taux de doublons (rafales vidéo/photo d'un même examen) et un bruit d'étiquetage notoire.
- Dans `src/data/cluster_patients.py`, le hachage perceptuel (`dHash + aHash`, Hamming $\le 6$) construit le graphe d'adjacence patient avec une condition bloquante :
  ```python
  # Ligne 166 de src/data/cluster_patients.py
  if labels[idx_i] == labels[idx_j]:
      row_ind.append(idx_i)
      col_ind.append(idx_j)
  ```
- Dans `config.yaml` : `use_weighted_sampler: false`, alors que les classes sont fortement asymétriques (~17% Type 1, ~53% Type 2, ~30% Type 3).
- Aucune dé-duplication physique n'est effectuée (les doublons sont conservés), ce qui sur-pondère les patientes ayant 10+ photos dans le gradient d'entraînement.

#### (b) Risque
- **Fuite de données résiduelle (Data Leakage) et Label Noise :** Si deux clichés quasi-identiques d'une même patiente ont reçu des étiquettes discordantes dans Kaggle (ex. image A annotée `Type_1`, image B annotée `Type_2`), la condition `if labels[idx_i] == labels[idx_j]` **ne les relie pas**. Elles reçoivent deux `patient_id` distincts et se retrouvent séparées entre Train et Val/Test.
- **Biais de prédiction vers la classe majoritaire :** Avec `use_weighted_sampler: false`, le modèle converge vers la classe `Type 2` au détriment de la classe `Type 1`.

#### (c) Recommandations Priorisées
1. **[P0 - Immédiat]** **Supprimer le filtrage par label dans le clustering :** Relier toutes les paires ayant une distance de Hamming $\le 6$. Si un cluster contient des labels contradictoires, **supprimer le cluster entier** (données corrompues) plutôt que de les injecter séparément.
2. **[P1]** **Dé-duplication active :** Ne conserver qu'une seule image (la plus nette via variance du Laplacien) par cluster patient pour assainir le signal d'entraînement.
3. **[P1]** **Activer l'échantillonnage équilibré :** Passer `use_weighted_sampler: true` ou utiliser une `Class-Balanced Cross-Entropy`.

---

### AXE 3 : Architecture à 2 Étages (CADe $\to$ CADx)

#### (a) Constat
- **CADe (Stage 1) :** Détecteur `yolov8n-det` configuré dans `config.yaml` (`img_size: 640`, `conf_threshold: 0.25`). La classe `JSCDetectorStage1` applique un crop avec 15% de marge ou un fallback `Center Crop 70%`.
- **CADx (Stage 2) :** `convnext_small` sur `img_size: [224, 224]`.
- **Déconnexion pratique dans le code :** `src/train.py` et `src/data/dataset.py` chargent les images brutes directement depuis `filepath`. **Le classifieur ConvNeXt s'entraîne sur l'image colposcopique entière, sans aucun crop YOLO préalable.**
- **Incohérence de résolution :** YOLO opère en 640, `crop_jsc` redimensionne en $384 \times 384$, et `IVADataset` / `config.yaml` redimensionnent en $224 \times 224$.

#### (b) Risque
- **Shortcut Learning (Apprentissage de raccourcis fallacieux) :** Sans découpage strict de la région cervicale, le réseau apprend la texture du spéculum (métallique vs plastique), la pilosité, les doigts de gants ou la marque de l'appareil photo plutôt que l'épithélium cervical.
- **Overfitting avec ConvNeXt-Small :** ConvNeXt-Small compte ~50M de paramètres. Sur un dataset de ~3 000 images réelles nettoyées, ce modèle est surdimensionné.
- **Incompatibilité Edge CSPS :** 50M de paramètres saturera la mémoire vive et videra la batterie d'un smartphone Android d'entrée de gamme en milieu rural.

#### (c) Recommandations Priorisées
1. **[P0]** **Brancher réellement le Stage 1 :** Pré-générer hors-ligne l'ensemble des crops ROI via YOLOv8n dans un dossier `data/processed/crops_yolo/` et entraîner le classifieur **strictement** sur ces vignettes rognées.
2. **[P1]** **Remplacer ConvNeXt-Small par une architecture compacte :** Adopter **`mobilenetv4_conv_medium`** (~9M params) ou **`efficientnet_b0`** (~5M params).
3. **[P1]** **Aligner la résolution à $256 \times 256$ ou $384 \times 384$ :** Le format 224x224 gomme les micro-motifs vasculaires (mosaïques et ponctuations fines) essentiels en colposcopie.

---

### AXE 4 : Augmentations & Simulation Réaliste (CMA/CSPS)

#### (a) Constat
- Dans `config.yaml` : `perlin_noise_proba: 0.15`, `specular_flare_proba: 0.15`, `hue_shift_limit: 0.05`.
- Dans `src/data/augmentations.py` : `FastPerlinNoiseLoader` injecte des masques 2D à seuil $> 0.6$ floutés, sous forme de patchs rouge vif ("blood") ou jaunâtres ("mucus") avec un `max_alpha: 0.4`.
- Pour les reflets spéculaires : utilisation de `A.RandomSunFlare` d'Albumentations.

#### (b) Risque
- **`RandomSunFlare` est une aberration optique médicale :** `RandomSunFlare` simule des rayons de soleil avec diffraction optique de paysage, alors qu'un reflet colposcopique est une zone de sur-exposition blanche pure ($V=255$), ponctuelle et circulaire/ovale, causée par le flash LED sur une surface humide.
- **Destruction de la sémantique de l'acéto-blanchiment :** Un masque de Perlin opaque ($\alpha=0.4$) ou un faux flare solaire non calibré recouvre la jonction squamo-columnaire et génère des gradients artificiels que le réseau peut assimiler à tort à une réaction acéto-blanche positive.
- **Aspect non-physiologique des masques de Perlin :** Les sécrétions biologiques (glaires cervicales, écoulements) suivent la gravité et l'anatomie du canal endocervical, pas une répartition fractale uniforme.

#### (c) Recommandations Priorisées
1. **[P0]** **Remplacer `RandomSunFlare` immédiatement :** Écrire une transformation dédiée simulant de petits îlots de saturation locale ($V \to 255$, $S \to 0$ dans l'espace HSV) avec bordures franches.
2. **[P1]** **Réduire l'intensité des masques de Perlin :** Abaisser `max_alpha` de 0.40 à **0.15** et restreindre l'application aux zones périphériques du col.
3. **[P2]** **Conserver le verrouillage chromatique :** Maintenir `hue_shift_limit <= 0.05` pour préserver la colorimétrie de l'acide acétique.

---

### AXE 5 : Métriques & Calibration

#### (a) Constat
- `config.yaml` définit `target_sensitivity: 0.95`, `min_specificity: 0.50`, `step: 0.01`.
- `eval_threshold_grid.py` balaye les seuils $T \in [0.05, 0.95]$ sur la somme des probabilités `probs[:, 1] + probs[:, 2]`.
- L'algorithme sélectionne le premier seuil qui atteint $95\%$ de sensibilité.

#### (b) Risque
- **Le piège du "Threshold Hacking" (Seuil Forcé) :** Si le modèle a un pouvoir discriminant médiocre (ex. AUC à 0.60), pour atteindre artificiellement $95\%$ de sensibilité, l'algorithme va effondrer le seuil de décision $T$ (ex. $T = 0.08$).
- **Spécificité effondrée sur le terrain :** Un seuil à $T=0.08$ conduit le système à déclarer 95% des patientes comme "positives / à référer". En milieu rural (CSPS), cela sature les structures secondaires et entraîne l'abandon de l'outil par le personnel soignant.
- **Déséquilibre de prévalence ignoré :** Avec une prévalence de lésions de haut grade de ~5% en population générale, un test à 95% de sensibilité et 50% de spécificité génère une **Valeur Prédictive Positive (VPP) dérisoire de ~9.1%** (plus de 9 alertes sur 10 sont de fausses alertes).

#### (c) Recommandations Priorisées
1. **[P0]** **Verrouiller la calibration sur un plancher d'AUC :** Interdire la sélection d'un seuil $T$ si l'AUC-ROC de validation est inférieure à **0.85**.
2. **[P1]** **Rapporter obligatoirement la VPP, VPN et l'ECE (Expected Calibration Error) :** Évaluer l'impact en fonction de la prévalence locale réelle au Burkina Faso.
3. **[P1]** **Adopter une zone d'incertitude (Triage à 3 bandes) :** Vert ($P < 0.20$), Jaune / Douteux ($0.20 \le P < 0.45 \to$ second badigeon acétique), Rouge ($P \ge 0.45$).

---

### AXE 6 : Entraînement & Hyperparamètres

#### (a) Constat
- Paramètres déclarés : `epochs: 20`, `warmup_epochs: 2`, `batch_size: 32`, `learning_rate: 1e-4` (backbone), `head_learning_rate: 1e-3`, `weight_decay: 1e-4`, `drop_rate: 0.2`, `label_smoothing: 0.05`.
- Dans `src/train.py` (l. 403) : La logique de checkpointing tente d'arbitrer entre `val_auc` (binaire) et `val_loss` (Cross-Entropy multiclasse).

#### (b) Risque
- **Durée d'entraînement trop courte :** Avec 2 époques de freeze (warmup) et 18 époques de dégel avec `CosineAnnealingLR`, le backbone dispose de moins de 20 cycles pour adapter ses filtres ImageNet aux textures subtiles de la muqueuse cervicale.
- **Conflit entre Label Smoothing et Seuil de Probabilité :** Le Label Smoothing (0.05) empêche mathématiquement les probabilités de sortie d'atteindre les extrêmes (0 et 1), ce qui biaise l'estimation des probabilités lors de la calibration fine du seuil $T$.
- **Critère de sauvegarde hybride instable :** Comparer une AUC binaire artificielle calculée sur des logits multiclasses avec une Cross-Entropy génère des sélections de checkpoints sous-optimaux.

#### (c) Recommandations Priorisées
1. **[P0]** **Unifier le critère de sélection de modèle :** Baser l'Early Stopping et la sauvegarde du meilleur modèle sur la **`val_macro_auc`** ou la **`val_loss` multiclasse pure**.
2. **[P1]** **Prolonger l'entraînement :** Passer à 35–40 époques avec `warmup_epochs: 3`.
3. **[P1]** **Désactiver le label smoothing lors de la phase de calibration :** Conserver la Cross-Entropy standard et appliquer un **Temperature Scaling (Calibration de Platt)** post-entraînement pour calibrer les probabilités.

---

### AXE 7 : Forces du Projet (À Conserver)

1. **Modularité et architecture logicielle soignée :** Structure du repository claire, scripts bien découpés (`src/data`, `src/models`, `src/utils`), configuration YAML centralisée.
2. **Differential Learning Rate :** Appliquer $10^{-4}$ sur le backbone et $10^{-3}$ sur la tête de classification avec une phase de dégel progressif (warmup) est la méthode optimale pour le transfer learning biomédical.
3. **Contrainte stricte sur le shift de teinte (`hue_shift <= 0.05`) :** Excellente décision clinique, la dérive colorimétrique artificielle étant un facteur majeur de dégradation des modèles en dermatologie et colposcopie.
4. **Intention de validation étanche (StratifiedGroupKFold) :** La prise en compte du risque de fuite par patient démontre une maturité méthodologique qu'il convient de parfaire.
5. **Orientation Edge AI :** Prévoir dès la conception l'export ONNX / quantification INT8 pour une exécution locale sans réseau sur smartphone.

---

### AXE 8 : Faiblesses & Risques Critiques (Top 5 par Gravité)

| Rang | Problème Identifié | Gravité | Impact | Correctif Immédiat |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Confusion ZT Type 1/2/3 $\leftrightarrow$ Cancer / CIN** | 🔴 **CRITIQUE** | Faux négatifs mortels sur cancers en ZT Type 1 ; Faux positifs massifs sur ZT Type 3 saines. | Recadrer le modèle en "Éligibilité au traitement ablatif" ; intégrer un dataset avec Gold Standard histologique pour la pathologie. |
| **2** | **Fuite de données résiduelle dans `cluster_patients.py`** | 🟠 **HAUTE** | Paires d'images d'une même patiente séparées entre Train et Val/Test si les labels Kaggle diffèrent. | Supprimer `if labels[idx_i] == labels[idx_j]` ; fusionner ou éliminer les clusters à labels contradictoires. |
| **3** | **CADe YOLO court-circuité à l'entraînement** | 🟠 **HAUTE** | ConvNeXt s'entraîne sur le spéculum et le vagin (Shortcut Learning massif). | Générer hors-ligne les crops ROI via YOLOv8n et entraîner ConvNeXt exclusivement sur les crops. |
| **4** | **Threshold Hacking (Sensibilité forcée à 95%)** | 🟡 **MOYENNE** | Spécificité effondrée ($< 20\%$) si l'AUC est faible, rendant l'outil inutilisable en CSPS. | Bloquer la calibration si $AUC < 0.85$ ; adopter un triage tri-bande (Vert / Jaune / Rouge). |
| **5** | **Augmentations irréalistes (`RandomSunFlare` & Perlin)** | 🟡 **MOYENNE** | Halos solaires et taches opaques perturbant l'apprentissage de l'acéto-blanchiment. | Remplacer par une saturation locale HSV (reflets de flash) et réduire l'opacité des masques. |

---

## 📊 Matrice Récapitulative des Risques

```
           GRAVITÉ
              ▲
   CRITIQUE   │                   [Risque 1: Confusion Clinique]
              │
      HAUTE   │         [Risque 2: Leakage Hash]   [Risque 3: YOLO déconnecté]
              │
    MOYENNE   │   [Risque 5: Augmentations]        [Risque 4: Threshold Hacking]
              │
              └─────────────────────────────────────────────────────────────► PROBABILITÉ
                       FAIBLE               MOYENNE               FORTE
```

| Risque | Gravité | Probabilité | Impact Projet / Clinique | Action Corrective |
| :--- | :---: | :---: | :--- | :--- |
| **1. Mésappariement sémantique des labels** | **Critique** | **Certaine (100%)** | Diagnostic erroné, mise en danger de patientes. | Découpler tâche anatomique et tâche de détection lésionnelle. |
| **2. Fuite patient via clustering conditionnel** | **Haute** | **Élevée (80%)** | Métriques de validation artificiellement gonflées. | Hachage non-conditionnel + purge des discordances. |
| **3. Non-utilisation des crops YOLO** | **Haute** | **Certaine (100%)** | Modèle sensible aux changements d'instruments et de cadre. | Pipeline d'entraînement 100% basé sur les crops. |
| **4. Seuil dégradé pour forcer le rappel** | **Moyenne** | **Élevée (75%)** | Explosion des faux positifs, saturation des CMA/CHU. | Conditionner le seuil à $AUC \ge 0.85$ + analyse VPP/VPN. |
| **5. Perturbations de texture non réalistes** | **Moyenne** | **Certaine (100%)** | Baisse de sensibilité sur les lésions réelles pâles. | Remplacer par des reflets spéculaires ponctuels HSV. |

---

## ❓ 10 Questions pour Validation avec les Experts Métier

### Questions Cliniques & Médicales (Gynécologues / OMS)
1. *Dans le protocole Screen-and-Treat en vigueur au Burkina Faso, confirmez-vous qu'une patiente avec ZT Type 3 sans lésion visible doit être référée en centre secondaire (CMA/CHU) pour bilan, alors qu'une ZT Type 1 ou 2 avec lésion éligible est traitée sur place par thermocoagulation ?*
2. *Quel est le taux maximal de faux positifs (spécificité minimale) acceptable par les sages-femmes en CSPS avant que le système ne devienne contre-productif ?*
3. *Quelle est la durée exacte observée entre l'application de l'acide acétique à 5% et l'apparition de l'acéto-blanchiment maximal dans les conditions thermiques des CSPS rurales ?*
4. *Existe-t-il une cohorte locale burkinabè (ex. CHU Yalgado Ouédraogo / Bogodogo) d'images VIA avec biopsies ou tests HPV appariés pour constituer un vrai jeu de test indépendant ?*

### Questions Machine Learning & Ingénierie
5. *Pourquoi le classifieur Stage 2 continue-t-il de charger les images brutes au lieu des boîtes englobantes issues du détecteur YOLOv8n ?*
6. *Dans `cluster_patients.py`, quelle est la justification mathématique de filtrer `if labels[idx_i] == labels[idx_j]` lors de la recherche de doublons d'un même patient ?*
7. *Quelle est la distribution réelle des classes après dé-duplication stricte sans condition de label ?*
8. *Pourquoi avoir choisi `convnext_small` (50M params) plutôt qu'un backbone mobile léger (`mobilenetv4` ou `efficientnet_b0`) pour une cible Android ARM64 ?*
9. *À quelle valeur chute la spécificité réelle lorsque le seuil $T$ est calibré pour garantir $95\%$ de sensibilité sur le jeu de validation ?*
10. *Quel est le comportement du modèle quantifié en INT8 face aux lésions acéto-blanches à très faible contraste ?*

---

## 🛠️ Snippets de Code Correctifs

### 1. Correction du Clustering Patient (Zéro Fuite & Purge du Bruit de Label)
Dans `src/data/cluster_patients.py` :
```python
# Correction Ligne 165+ : Relier TOUTES les images similaires sans condition de classe
for r, c in zip(match_r, match_c):
    idx_i = i + r
    idx_j = j + c
    if idx_i != idx_j:
        # Relier inconditionnellement pour capturer toutes les photos d'un même patient
        row_ind.append(idx_i)
        col_ind.append(idx_j)

# Après calcul des composantes connexes : Purge des clusters avec étiquettes contradictoires
df_clusters = pd.DataFrame({'filepath': image_paths, 'label': labels, 'patient_id': patient_ids})
cluster_label_counts = df_clusters.groupby('patient_id')['label'].nunique()
corrupted_patients = cluster_label_counts[cluster_label_counts > 1].index

print(f"⚠️ {len(corrupted_patients)} clusters patients présentent des étiquettes discordantes (bruit de label).")
# Élimination des clusters corrompus pour préserver la vérité terrain
df_clean = df_clusters[~df_clusters['patient_id'].isin(corrupted_patients)].copy()
```

---

### 2. Échantillonnage Pondéré (Gestion du Déséquilibre de Classes)
Dans `src/train.py` :
```python
from torch.utils.data import WeightedRandomSampler

# Calcul des poids inversement proportionnels à la fréquence de classe
train_targets = train_dataset.df['target'].values
class_counts = np.bincount(train_targets, minlength=3)
class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
# Normalisation des poids par classe
class_weights = class_weights / class_weights.sum()

sample_weights = class_weights[train_targets]
sampler = WeightedRandomSampler(
    weights=torch.as_tensor(sample_weights, dtype=torch.double),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler, # Remplacement de shuffle=True par le sampler
    num_workers=num_workers,
    pin_memory=pin_memory
)
```

---

### 3. Calibration de Seuil Sécurisée avec Plancher d'AUC & Triage Tri-Bande
Dans `src/utils/metrics.py` :
```python
def calibrate_clinical_threshold_safe(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_auc_threshold: float = 0.85,
    target_sens: float = 0.95,
    min_spec: float = 0.50
) -> dict:
    """
    Calibration sécurisée du seuil décisionnel interdisant le threshold hacking
    si le modèle sous-jacent manque de pouvoir discriminant.
    """
    from sklearn.metrics import roc_auc_score
    
    auc = roc_auc_score(y_true, y_prob)
    if auc < min_auc_threshold:
        return {
            "status": "REJECTED_IMMATURE_MODEL",
            "auc_roc": float(auc),
            "recommended_threshold": 0.50,
            "warning": f"AUC ({auc:.3f}) < {min_auc_threshold}. Modèle non calibrable en sécurité."
        }

    best_t = 0.50
    best_spec = 0.0
    
    for t in np.arange(0.10, 0.90, 0.01):
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        
        if sens >= target_sens and spec >= min_spec:
            if spec > best_spec:
                best_spec = spec
                best_t = float(t)
                
    return {
        "status": "OPTIMAL_CALIBRATED",
        "auc_roc": float(auc),
        "optimal_threshold": best_t,
        "achieved_specificity": float(best_spec)
    }
```

---

### 4. Transformateur Réaliste de Reflets Spéculaires (Flash LED sur Muqueuse)
Dans `src/data/augmentations.py` :
```python
class SpecularFlashReflectance:
    """
    Simule fidèlement les reflets spéculaires du flash LED d'un smartphone
    sur une muqueuse cervicale humidifiée (zones saturées V=255, S=0 à contour net).
    """
    def __init__(self, p: float = 0.3, max_reflections: int = 3):
        self.p = p
        self.max_reflections = max_reflections

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return image
        
        h, w, _ = image.shape
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        num_flares = np.random.randint(1, self.max_reflections + 1)
        for _ in range(num_flares):
            cx = np.random.randint(int(w * 0.2), int(w * 0.8))
            cy = np.random.randint(int(h * 0.2), int(h * 0.8))
            axes = (np.random.randint(4, 15), np.random.randint(3, 10))
            angle = np.random.randint(0, 180)
            
            # Création du spot spéculaire saturé
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
            
            # Saturation blanche pure au centre, liseré net
            img_hsv[mask == 255, 1] = np.random.randint(0, 30)   # Sature vers le blanc
            img_hsv[mask == 255, 2] = np.random.randint(245, 256) # Luminance maximale
            
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
```
