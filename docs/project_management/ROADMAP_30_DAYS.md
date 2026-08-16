# 📅 ROADMAP_30_DAYS.md — Feuille de Route d'Ingénierie & Déploiement Edge AI (LAAFI_AI_VIA)

> **Document d'Exécution Projet (v2.0)**  
> **Objectif :** Faire passer le moteur **LAAFI_AI_VIA** d'un prototype R&D à un modèle *Edge AI* de production clinique pour smartphones Android en zone rurale au Burkina Faso.  
> **Cible Clinique :** Sensibilité $\ge 95.0\%$ (priorité absolue), Spécificité $\ge 85.0\%$, Score $F_2 \ge 0.93$.  
> **Cible Matérielle :** Inférence 100% hors-ligne, résolution $384 \times 384$, poids INT8 $\le 15\text{ MB}$, latence $< 250\text{ ms}$, RAM $< 150\text{ MB}$.

---

## 🗓️ 1. Vue d'Ensemble du Planning (30 Jours / 4 Semaines)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PLANNING D'EXÉCUTION 30 JOURS                         │
├──────────────┬──────────────────────────────────────────────────────────────┤
│ Semaine 1    │ Preprocessing Reflets-Lite & Ablation Loss (FP32 Baseline)  │
│ Semaine 2    │ Distillation Hybride (Teacher ConvNeXt ➔ Student MobileNet)  │
│ Semaine 3    │ Export Graph, PTQ / QAT INT8 & Quantification ExecuTorch    │
│ Semaine 4    │ Validation Android ARM64, Benchmarks Edge & Hardening        │
└──────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 📅 2. Programme Détaillé Jour par Jour

### 🗓️ SEMAINE 1 : Preprocessing Reflets-Lite & Optimisation des Loss (FP32)
* **Objectif :** Atteindre $\ge 85.0\%$ de Spécificité à Sensibilité $\ge 95.0\%$ sur le modèle FP32.

* **Jour 1 — Module Reflets-Lite (`src/preprocessing/cervix_transforms.py`) :**
  - Implémenter le masquage HSV binaire sans inpainting lourd.
  - Remplacer les pixels saturés par la couleur moyenne locale du col (évite les trous noirs à fort contraste).
* **Jour 2 — Loss Asymétrique (`src/losses/asymmetric_loss.py`) :**
  - Implémenter `AsymmetricFocalLoss` ($\gamma_{\text{pos}}=1.0, \gamma_{\text{neg}}=4.0$) compatible avec logit binaire `[B, 1]`.
* **Jours 3-4 — Script d'Ablation (`experiments/run_ablation.py`) :**
  - Lancer la grille d'expérimentation comparant : `FocalLoss` ($\alpha=0.75$) vs `AsymmetricFocalLoss` vs Reflets-Lite.
* **Jour 5 — Calibration du Seuil $T_{\text{opt}}$ :**
  - Implémenter la recherche du seuil $T_{\text{opt}} \in [0.10, 0.90]$ **exclusivement sur `val.csv`** avec la contrainte $\text{Recall} \ge 95.0\%$.
* **Jours 6-7 — Validation Aveugle :**
  - Geler $T_{\text{opt}}$, évaluer sur `test.csv` (1 682 images) et valider la matrice de confusion.

---

### 🗓️ SEMAINE 2 : Distillation de Connaissances (Teacher ➔ Student)
* **Objectif :** Transférer la performance de `ConvNeXt-Base` vers `MobileNetV4-Small` ($384 \times 384$).

* **Jour 8 — Gel du Teacher :**
  - Charger les poids du Teacher `ConvNeXt-Base` et passer en mode `eval()` sans écriture disque volumineuse.
* **Jours 9-10 — Loss KD Binaire & Attention Transfer (`src/distillation/kd_loss.py`) :**
  - Implémenter `BinaryHybridKDLoss` avec Soft-BCE ($T=4.0$) et alignement d'attention spatiale via `AdaptiveAvgPool2d`.
* **Jours 11-12 — Entraînement du Student :**
  - Entraîner `MobileNetV4-Small` (via `timm`, $384 \times 384$) avec schedule `CosineAnnealingLR` et LLRD ($LR_{\text{backbone}}=10^{-5}, LR_{\text{head}}=10^{-4}$).
* **Jours 13-14 — Évaluation FP32 Student :**
  - Vérifier que le Student FP32 conserve $\ge 98\%$ des métriques du Teacher (Sensibilité $\ge 95\%$, Spécificité $\ge 85\%$).

---

### 🗓️ SEMAINE 3 : Quantification INT8 & Toolchain ExecuTorch
* **Objectif :** Compresser le modèle sous $15\text{ MB}$ sans perte clinique ($< 0.5\%$).

* **Jours 15-16 — Export Graph `torch.export` (`export/export_executorch.py`) :**
  - Exporter le graphe PyTorch avec entrée explicite `(1, 3, 384, 384)`.
* **Jours 17-18 — Post-Training Quantization (PTQ INT8) :**
  - Calibrer le modèle INT8 avec 500 images représentatives via ONNX Runtime / ExecuTorch.
* **Jour 19 — Quantization Gate Check :**
  - Évaluer le modèle quantisé INT8 sur `test.csv`.
  - *Si perte métrique $< 0.5\%$* ➔ Validation PTQ.
  - *Si perte métrique $\ge 0.5\%$* ➔ Déclenchement du QAT.
* **Jours 20-21 — (Fallback QAT) :**
  - Ré-entraîner le Student 3 à 5 époques avec insertion de fake-quantization nodes.

---

### 🗓️ SEMAINE 4 : Validation Android Edge AI & Qualification
* **Objectif :** Valider l'exécution 100% hors-ligne sur processeurs ARM64.

* **Jours 22-23 — Integration Android Harness :**
  - Intégrer le binaire `.pte` / `.onnx` dans le runner C++/Java Android.
* **Jours 24-25 — Benchmarks Matériels (Snapdragon / Helio G88) :**
  - Mesurer : Latence Preprocessing ($< 2\text{ ms}$), Latence Inférence ($< 150\text{ ms}$), Allocation RAM ($< 150\text{ MB}$).
* **Jours 26-27 — Tests de Robustesse :**
  - Tester la tenue aux bruits de terrain (faible éclairage, flou de bougé, reflets sévères).
* **Jours 28-30 — Release Packaging :**
  - Geler la version du modèle, générer le manifeste de release et archiver le code de production.

---

## 📊 3. Grille d'Acceptation Release (Gate Opérationnel)

| Critère de Contrôle | Seuil Strict Pass/Fail | Méthode de Validation |
| :--- | :--- | :--- |
| **Sensibilité Clinique** | **$\ge 95.0\%$** | Test aveugle sur `test.csv` (1 682 images) |
| **Spécificité Clinique** | **$\ge 85.0\%$** | Test aveugle sur `test.csv` avec $T_{\text{opt}}$ gelé |
| **Poids Modèle INT8** | **$\le 15 \text{ MB}$** | Taille du fichier `.pte` / `.onnx` |
| **Latence CPU ARM** | **$\le 250 \text{ ms}$** | Benchmark sur Snapdragon 680 / Helio G88 |
| **Delta Quantification** | **$< 0.5\%$** | Dégradation relative FP32 vs INT8 |
