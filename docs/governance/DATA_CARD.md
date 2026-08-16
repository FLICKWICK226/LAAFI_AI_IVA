# 🗂️ DATA CARD — Jeu de Données LAAFI_AI_IVA

## 1. Origine & Provenance des Données
- **Source Primaire :** *Intel & MobileODT Cervical Cancer Screening Dataset* (Kaggle public dataset).
- **Volume Total :** 8 215 clichés numériques haute résolution de col utérin après application d'acide acétique.
- **Modalité :** Photographies smartphone et colposcope mobile en lumière blanche avec flash LED.

---

## 2. Découpage & Stratégie Anti-Data Leakage
- **Problématique :** Présence de séries temporelles de plusieurs clichés pris sur une même patiente lors d'une même séance d'examen.
- **Méthode de Split :** `StratifiedGroupKFold(n_splits=5, shuffle=True, seed=42)` avec `groups = patient_id`.
- **Répartition :**
  - **Train Set :** $70\%$ des patientes ($\approx 5\,750$ images)
  - **Validation Set :** $15\%$ des patientes ($\approx 1\,232$ images) — Réservé pour la calibration du seuil $T_{\text{opt}}$.
  - **Test Set :** $15\%$ des patientes ($\approx 1\,233$ images) — Évaluation aveugle finale.

---

## 3. Prétraitements & Données Synthétiques
- **Génération de Masques Synthétiques :** 1 000 masques procéduraux de bruit de Perlin pour simuler le mucus et les saignements mineurs.
- **Résolution Standardisée :** Crops carrés $384 \times 384$ pixels centrés sur la Jonction Squamo-Columnaire (JSC).
