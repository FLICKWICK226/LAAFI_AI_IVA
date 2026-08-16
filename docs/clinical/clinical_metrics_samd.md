# 📊 Métriques Cliniques & Exigences Réglementaires SaMD

> **Dispositif Médical Logiciel (SaMD) :** LAAFI_AI_IVA Engine v2.0  
> **Classification Réglementaire Cible :** Classe IIa / IIb (CE MDR 2017/745 / FDA CADe/CADx)

---

## 1. Grille des Objectifs Cliniques

| Métrique Clinique | Baseline R&D | Cible Production | Justification Médicale & Éthique |
| :--- | :---: | :---: | :--- |
| **Sensibilité (Recall)** | **$\ge 95.0\%$** | **$\ge 97.0\%$** | **Sécurité Patient Absolue :** Un Faux Négatif est une patiente renvoyée chez elle avec un précancer évolutif non traité. |
| **Spécificité** | **$\ge 80.0\%$** | **$\ge 85.0\%$** | **Efficience Système :** Réduire les Faux Positifs pour éviter l'ablation thermique inutile et l'engorgement des CMA/CHU. |
| **Score $F_2$** | **$\ge 0.88$** | **$\ge 0.93$** | Métrique asymétrique officielle : $\beta=2$ accorde un poids double au Recall par rapport à la Précision. |
| **AUC-ROC** | **$\ge 0.90$** | **$\ge 0.94$** | Capacité intrinsèque globale de discrimination du modèle indépendamment du seuil de décision. |

---

## 2. Formulation Mathématique du Score $F_\beta$ ($\beta=2$)

$$F_2 = (1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(2^2 \cdot \text{Precision}) + \text{Recall}} = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$

---

## 3. Règle de Calibration du Seuil Décisionnel ($T_{\text{opt}}$)

1. L'optimisation du seuil est effectuée **exclusivement sur l'ensemble de validation (`val.csv`)**.
2. On balaie $T \in [0.10, 0.90]$ avec un pas de $0.01$.
3. On sélectionne le seuil $T_{\text{opt}}$ qui maximise la spécificité **sous la contrainte stricte que $\text{Recall}(T) \ge 95.0\%$**.
4. Le seuil $T_{\text{opt}}$ est ensuite **gelé** et appliqué à l'aveugle sur le jeu de test indépendant (`test.csv`).
