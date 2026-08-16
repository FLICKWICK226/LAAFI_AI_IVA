# 📇 MODEL CARD — LAAFI_AI_IVA Engine v2.0

## 1. Informations Générales
- **Nom du Modèle :** LAAFI_AI_IVA Engine (Stage 2 Classifier)
- **Version :** 2.0.0
- **Type d'Architecture :**
  - **Teacher :** ConvNeXt-Base (88M params)
  - **Student :** MobileNetV4-Small (3.8 MB INT8)
- **Date de Release :** Août 2026
- **Auteurs :** LAAFI_AI Team

---

## 2. Usage Prévu & Limites d'Utilisation
- **Usage Clinique Prévu :** Dispositif d'aide à la décision (CADx) pour le triage des patientes lors du dépistage du cancer du col de l'utérus par acide acétique (IVA/VIA) en soins primaires (CSPS).
- **Utilisateurs Cibles :** Sages-femmes, infirmiers d'état et médecins généralistes formés à l'IVA.
- **Contre-indications :** Ne doit pas être utilisé comme diagnostic histopathologique définitif ; ne se substitue pas à la biopsie en cas de col macroscopiquement suspect d'invasion maligne.

---

## 3. Performances Cliniques (Test Set Indépendant - 1 682 images)
- **Sensibilité (Recall CIN2+) :** $95.0\%$
- **Spécificité :** $93.3\%$
- **Score $F_2$ :** $0.926$
- **AUC-ROC :** $0.9293$

---

## 4. Considérations Éthiques & Biais
- **Validation Inter-Capteurs :** Risque de biais de balance des blancs et température de flash LED selon la marque de smartphone.
- **Atténuation :** Prétraitement avec le module `Reflets-Lite` et augmentations photométriques robustes (décalage HSV, contraste local adaptatif).
