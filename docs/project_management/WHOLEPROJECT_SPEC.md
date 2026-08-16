# 🔬 WHOLEPROJECT_SPEC.md — Architecture, Spécifications Cliniques & Feuille de Route SOTA (LAAFI_AI_VIA)

> **Document de Référence Technique & Brief de Recherche Deep Search**  
> **Projet :** LAAFI_AI_VIA Engine (Version 2.0)  
> **Domaine :** Dispositif Médical Logiciel (SaMD) CADe/CADx pour le dépistage du Cancer du Col de l'Utérus par Imagerie Smartphone (IVA/VIA)  
> **Cible Terrain :** Centres de Santé et de Promotion Sociale (CSPS) en zones rurales au Burkina Faso / Afrique Subsaharienne  

---

## 📋 1. Executive Summary & Contextualisation Clinique

### 1.1 Vision du Projet & Problématique Terrain

Le cancer du col de l'utérus est le 4ème cancer le plus fréquent chez la femme au niveau mondial, mais **plus de 90% des décès surviennent dans les pays à revenus faibles ou intermédiaires (LMIC)** comme le Burkina Faso.
En zone rurale (CSPS, CM, CMA), l'absence de colposcopes, de cytologie (Frottis) et de laboratoires d'anatomopathologie rend la méthode **"Screen-and-Treat" (Dépister et Traiter)** basée sur l'Inspection Visuelle à l'Acide Acétique (IVA/VIA) obligatoire selon les recommandations de l'OMS.

L'IVA traditionnelle présente toutefois une variabilité inter-observateur critique (sensibilité de 45% à 85% selon l'expérience des infirmiers/sage-femmes). **LAAFI_AI_VIA** est une solution *Edge AI* de Computer Vision intégrée sur smartphone Android, fonctionnant **100% hors-ligne**, fournissant une assistance au diagnostic en temps réel.

### 1.2 Grille des Métriques Cliniques Cibles (SaMD Class IIa/IIb)

| Métrique Clinique | Seuil Minimal Acceptable (Baseline) | Seuil Cible (Production SOTA) | Justification Médicale & Éthique |
| :--- | :--- | :--- | :--- |
| **Sensibilité (Recall)** | **$\ge 95.0\%$** | **$\ge 97.0\%$** | **Priorité Absolue :** Minimiser à zéro les Faux Négatifs (lésions précancéreuses CIN2/CIN3 ratées). |
| **Spécificité** | **$\ge 80.0\%$** | **$\ge 85.0\%$** | **Efficacité Système :** Réduire les Faux Positifs pour éviter les actes d'ablation thermique (thermocoagulation) inutiles et les frais de transport des familles rurales vers les CHU. |
| **Score $F_2$** | **$\ge 0.88$** | **$\ge 0.93$** | Métrique de référence accordant un poids $2\times$ supérieur au Recall par rapport à la Précision. |
| **AUC-ROC** | **$\ge 0.90$** | **$\ge 0.94$** | Capacité globale de discrimination binaire et tri-classe. |

### 1.3 Contraintes Matérielles & Déploiement Edge AI

| Contrainte Matérielle | Seuil Strict (Edge Production) | Justification Terrain (Burkina Faso) |
| :--- | :--- | :--- |
| **Mode d'Exécution** | **100% Hors-Ligne (Offline)** | Aucune dépendance au réseau 3G/4G/Cloud dans les CSPS ruraux. |
| **Cible Matérielle** | Smartphone ARM64 (4–6 Go RAM) | Ex: Xiaomi Redmi, Samsung Galaxy A (SOC Octa-core ARM). |
| **Poids Binaire Modèle** | **$\le 25 \text{ MB}$** | Stockage limité, déploiement APK rapide via téléversement local. |
| **Latence CPU ARM** | **$\le 250 \text{ ms}$ / image** | Retour d'information immédiat pendant l'examen de 1 minute. |
| **Allocation RAM** | **$\le 150 \text{ MB}$** | Prévenir tout plantage (*OutOfMemory*) de l'application Android. |
| **Résolution d'Entrée** | **$384 \times 384$ pixels** | Préservation obligatoire des détails fins de la JSC (ponctuations, vascularisation acéto-blanche). |

---

## 🏗️ 2. Architecture Actuelle & Baseline R&D

### 2.1 Pipeline à Deux Étages (CADe / CADx)

1. **Stage 1 (CADe — Localisation & Crop ROI JSC) :**
   - **Modèle :** `YOLOv8n-det` ou `YOLOv11n-det` ($\sim 3.2 \text{ MB}$ FP32).
   - **Rôle :** Détecter la Zone de Jonction Squamo-Columnaire (JSC) sur le cliché natif, isoler le col et rogner la ROI ($384 \times 384$) pour éliminer $>70\%$ d'arrière-plan parasite (spéculum, parois vaginales, gants).
2. **Stage 2 (CADx — Classification Lésionnelle & Éligibilité) :**
   - **Backbone R&D (Teacher) :** `ConvNeXt-Base` (88M paramètres, pré-entraîné ImageNet-1k).
   - **Entrée :** Crop JSC $384 \times 384 \times 3$.
   - **Sortie :** Logits de classification binaire (Sain vs VIA-Positif) + Triage Tri-Classe.

### 2.2 Stratégie de Validation Anti-Data Leakage

- **Problématique :** Dans le dataset *Intel-MobileODT* (8 215 images indexées), plusieurs clichés appartiennent à la même patiente (séries temporelles post-acide acétique). Un split aléatoire naïf provoquerait un *patient leakage* massif.
- **Solution Implémentée :**
  - Extraction systématique de `patient_id` sur chaque image.
  - Découpage par **`StratifiedGroupKFold(n_splits=5, shuffle=True, seed=42)`** avec `groups = patient_id`.
  - **Garantie :** 100% des clichés d'une même patiente restent hermétiquement confinés dans un seul split (`train`, `val` ou `test`).

### 2.3 Traitement de l'Asymétrie & Calibration

- **Perte Asymétrique :** `FocalLoss` avec $\gamma = 2.0$ et $\alpha = 0.75$ pour pénaliser $3\times$ plus fort les Faux Négatifs.
- **Seuil Dynamique ($T_{\text{opt}}$) :** Balayage fin $T \in [0.10, 0.90]$ par pas de $0.01$ effectué **exclusivement sur la validation (`val.csv`)**. Le seuil optimisé qui garantit $\text{Recall} \ge 95.0\%$ tout en maximisant la spécificité est gelé, puis évalué à aveugle sur le test set indépendant (`test.csv`).
- **Résultats Test Set Actuels :** **Sensibilité = 95.0%**, **Spécificité = 93.3%**, **AUC-ROC = 0.9293**, **$F_2$ = 0.926**.

---

## 🚀 3. Feuille de Route d'Optimisation & Entraînement Dynamique

### 3.1 Corrections Méthodologiques de l'Entraînement

1. **Suppression des 30 Époques Fixes ➔ Early Stopping Basé sur `val_auc` / `val_f2` :**
   - Interdiction d'un nombre fixe d'époques. Max epochs = 30 avec `Patience = 5-6`.
   - L'Early Stopping est piloté par l'**`AUC-ROC` de Validation** (métrique intrinsèque indépendante du seuil) ou le **$F_2$ au seuil calibré**, évitant l'arrêt prématuré causé par les oscillations de seuil.
2. **Dégel Progressif du Backbone (Layer-wise Learning Rate Decay - LLRD) :**
   - Remplacement du dégel brutal à l'époque 4.
   - Application d'un dégel progressif avec taux d'apprentissage différencié : $LR_{\text{backbone}} = 10^{-5}$ et $LR_{\text{head}} = 10^{-4}$, associé à un `CosineAnnealingLR` ($LR_{\text{min}} = 10^{-6}$).
3. **Pondération Rééquilibrée $\alpha=0.50$ :**
   - Ajustement de $\alpha$ de $0.75$ à $0.50-0.60$ dans la Focal Loss pour éviter d'écraser la spécificité lors des premières époques.

### 3.2 Pipeline de Distillation de Connaissances (Teacher ➔ Student)

Pour passer du modèle R&D `ConvNeXt-Base` ($340 \text{ MB}$) au binaire Edge ($< 25 \text{ MB}$) sans perte clinique :

```text
[ Teacher: ConvNeXt-Base (88M params / 340 MB) ] ──► (Génère Logits Lisses T=4.0)
                                 │
                                 ▼ Knowledge Distillation (KD)
[ Student: MobileNetV4-Small / EfficientNet-B0 (~12 MB FP32) ]
                                 │
                                 ▼ Post-Training Quantization (PTQ INT8)
[ Final Edge Model: ONNX Runtime / ExecuTorch (~3.5 MB INT8) ]
```

- **Perte de Distillation Combinée :**
$$\mathcal{L}_{\text{total}} = (1 - \alpha_{\text{kd}}) \cdot \mathcal{L}_{\text{Focal}}(y, \hat{y}_{\text{student}}, \alpha=0.50) + \alpha_{\text{kd}} \cdot T_{\text{temp}}^2 \cdot \mathcal{L}_{\text{KL}}\left(\sigma\left(\frac{z_{\text{student}}}{T_{\text{temp}}}\right), \sigma\left(\frac{z_{\text{teacher}}}{T_{\text{temp}}}\right)\right)$$
*(Hyperparamètres : $\alpha_{\text{kd}} = 0.6$, $T_{\text{temp}} = 4.0$)*

---

## 🔍 4. Axes de Deep Search (Brief de Recherche SOTA)

Cette section définit les **8 axes prioritaires de recherche bibliographique et d'exploration scientifique** pour propulser l'état de l'art du moteur LAAFI_AI_VIA :

### Axis 1 : Self-Supervised Pretraining (SSL) sur Imagerie Gynécologique / IVA
- **Sujet :** L'entraînement pré-requis sur ImageNet présente un biais de domaine fort par rapport à l'imagerie muqueuse cervicale.
- **Mots-clés Deep Search :** `Self-Supervised Learning cervical VIA`, `DINOv2 fine-tuning medical colposcopy`, `SimCLR medical endoscopy`, `Masked Autoencoders (MAE) cervical lesions`.
- **Objectif :** Évaluer si un pré-entraînement SSL non supervisé sur des milliers d'images VIA brutes améliore la séparation des classes et la spécificité (+3 à +5%).

### Axis 2 : Loss Functions Spécifiques pour Asymétrie Clinique & Classes Déséquilibrées
- **Sujet :** Optimisation de la fonction d'objectif au-delà de la Focal Loss classique.
- **Mots-clés Deep Search :** `Asymmetric Loss (ASL) for medical classification`, `AUC-ROC direct maximization loss`, `Dice-Focal hybrid loss binary medical`, `Cost-sensitive learning false negative penalty`.
- **Objectif :** Tester des pertes optimisant directement l'aire sous la courbe ROC ou la pénalité asymétrique non linéaire pour stabiliser la spécificité au-dessus de 88%.

### Axis 3 : Test-Time Augmentation (TTA) Ultra-Léger pour Inférence Edge
- **Sujet :** Améliorer la robustesse des prédictions sur smartphone lors des prises de vue de mauvaise qualité sans exploser la latence.
- **Mots-clés Deep Search :** `Efficient Test-Time Augmentation Edge AI`, `Low-latency TTA smartphone inference`, `Multi-crop geometric TTA medical mobile`.
- **Objectif :** Concevoir une stratégie TTA à 2 ou 3 flips/rotations légères en $< 80 \text{ ms}$ supplémentaires pour lisser les incertitudes de prédiction.

### Axis 4 : Attention-Guided Feature Distillation (KD Attention-Based)
- **Sujet :** Transférer non seulement les logits de sortie du Teacher vers le Student, mais également les cartes d'attention visuelle sur la JSC.
- **Mots-clés Deep Search :** `Attention Transfer Knowledge Distillation medical`, `Feature-based distillation ConvNeXt to MobileNet`, `Grad-CAM guided distillation loss`.
- **Objectif :** Forcer le modèle Student léger (`MobileNetV4`) à focaliser son attention sur les mêmes zones vasculaires et acéto-blanches que le Teacher `ConvNeXt-Base`.

### Axis 5 : Adaptation de Domaine & Robustesse aux Caméras Smartphones (Domain Generalization)
- **Sujet :** Variabilité de capteurs CMOS, de température de couleur du flash LED et d'optiques entre différentes marques de smartphones (Xiaomi vs Samsung vs Tecno).
- **Mots-clés Deep Search :** `Domain Generalization smartphone medical imaging`, `Color constancy algorithm VIA acetic acid`, `Stain normalization cervical VIA`, `StyleAugment mobile medical vision`.
- **Objectif :** Garantir que les performances (Sensibilité 95%+, Spécificité 85%+) ne chutent pas lors du passage d'un modèle de smartphone à un autre.

### Axis 6 : Multi-Task Learning & Joint Eligibility/Diagnosis Architecture
- **Sujet :** Prédire conjointement l'éligibilité anatomique (Type 1/2 vs Type 3) et le diagnostic lésionnel (Sain vs VIA+) au sein d'une seule tête multi-tâche.
- **Mots-clés Deep Search :** `Multi-task learning cervical VIA classification`, `Joint eligibility and lesion diagnosis CNN`, `Hard parameter sharing medical vision`.
- **Objectif :** Mutualiser les représentations de caractéristiques et réduire la taille totale du modèle par rapport à deux réseaux séparés.

### Axis 7 : Quantification Améliorée (QAT - Quantization-Aware Training) vs PTQ
- **Sujet :** Éviter toute perte de sensibilité lors du passage de FP32 à INT8 sur le processeur mobile ARM.
- **Mots-clés Deep Search :** `Quantization-Aware Training PyTorch mobile`, `INT8 quantization accuracy drop medical CNN`, `ONNX Runtime INT8 calibration medical`.
- **Objectif :** Comparer la quantification PTQ avec la quantification QAT pour garantir un delta de sensibilité $< 0.5\%$ après conversion 8-bit.

### Axis 8 : Triage Tri-Classe & Calibration des Probabilités (Platt Scaling / Temperature Scaling)
- **Sujet :** Régulariser et calibrer les probabilités de sortie du modèle pour le système de triage Vert / Jaune / Rouge.
- **Mots-clés Deep Search :** `Probability calibration medical classification`, `Temperature scaling binary decision`, `Tri-class risk triage thresholding`.
- **Objectif :** Garantir que la probabilité prédite $P$ corresponde exactement au risque clinique réel de lésion précancéreuse CIN2+.
