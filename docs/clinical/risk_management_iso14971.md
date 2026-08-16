# 🛡️ Matrice de Gestion des Risques Cliniques (ISO 14971)

> **Dispositif Médical Logiciel (SaMD) :** LAAFI_AI_IVA Engine v2.0  
> **Norme Référente :** ISO 14971 (Application de la gestion des risques aux dispositifs médicaux)

---

## Matrice des Dangers, Conséquences et Mesures d'Atténuation

| ID Danger | Événement Déclencheur | Conséquence Clinique | Niveau de Gravité Initial | Mesure d'Atténuation Implémentée dans LAAFI_AI | Niveau de Gravité Résiduel |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **HAZ-01** | Faux Négatif (Lésion CIN2/CIN3 ratée par l'IA). | Évolution non détectée vers un cancer invasif du col. | **Critique** | • Optimisation du seuil décisionnel pour garantir $\text{Sensibilité} \ge 95.0\%$.<br>• Fonction de perte `AsymmetricFocalLoss` pénalisant $4\times$ les Faux Négatifs.<br>• Système de Triage Tri-classe (Zone Jaune pour 2nd avis). | **Faible** |
| **HAZ-02** | Faux Positif (Col sain diagnostiqué positif). | Thermocoagulation inutile, anxiété, déplacement coûteux vers un CHU. | **Modéré** | • Stage 1 YOLO éliminant les reflets de spéculum et parois vaginales.<br>• Filtre de prétraitement `Reflets-Lite` masquant les reflets LED parasites. | **Faible** |
| **HAZ-03** | Image floue ou mauvaise exposition photographique. | Prédiction erronée due à la dégradation de l'image. | **Majeur** | • Entraînement avec augmentations de flou de bougé et masques de Perlin.<br>• Contrôle qualité automatisé avant inférence. | **Faible** |
| **HAZ-04** | Fuite de données entre patientes (Patient Leakage). | Surestimation artificielle des performances du modèle. | **Critique** | • Découpage strict par `StratifiedGroupKFold` basé sur l'identifiant patient (`patient_id`). | **Négligeable** |
| **HAZ-05** | Dégradation de précision post-quantification INT8. | Chute de sensibilité sur le smartphone ARM64. | **Majeur** | • Quantization Gate Check imposant $\Delta \text{Sensibilité} < 0.5\%$.<br>• Fallback automatisé sur QAT (Quantization-Aware Training) si échec PTQ. | **Faible** |
