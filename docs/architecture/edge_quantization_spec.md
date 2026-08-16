# ⚡ Spécifications de Quantification Edge AI (PTQ / QAT)

> **Moteur :** LAAFI_AI_IVA Engine v2.0  
> **Cible Matérielle :** Processeurs ARM64 (Snapdragon, MediaTek Helio/Dimensity) sans connexion Internet.

---

## 1. Contraintes Matérielles & Métriques Cibles

| Paramètre | Seuil Strict | Justification Terrain (Burkina Faso) |
| :--- | :---: | :--- |
| **Poids du Modèle** | **$\le 15\text{ MB}$ (INT8)** | Téléversement direct dans l'APK Android et stockage minimal. |
| **Latence CPU ARM** | **$\le 250\text{ ms}$ / image** | Retour diagnostique instantané pendant la consultation. |
| **Consommation RAM** | **$\le 150\text{ MB}$** | Éviter tout plantage OOM (Out Of Memory) sur les téléphones à 4 Go RAM. |
| **Perte Clinique ($\Delta$ Sensibilité)** | **$< 0.5\%$** | Préservation intégrale de la sécurité patient post-quantification. |

---

## 2. Stratégie de Déploiement en Deux Étapes

```text
[ Modèle Student MobileNetV4 FP32 (~15 MB) ]
                     │
                     ▼
[ Étape 1 : Post-Training Quantization (PTQ) ] ── (Calibration 500 images réelles)
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
[ Gate Check Validé ]   [ Gate Check Échoué ]
(Δ Sensibilité < 0.5%)  (Δ Sensibilité >= 0.5%)
         │                       │
         │                       ▼
         │           [ Étape 2 : QAT Fallback ] ── (Fine-tuning 3-5 époques avec FakeQuant)
         │                       │
         └───────────┬───────────┘
                     ▼
[ Modèle Final INT8 : ONNX Runtime Mobile / ExecuTorch (~3.8 MB) ]
```
