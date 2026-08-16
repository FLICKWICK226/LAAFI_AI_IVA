# 🧠 Cadre de Distillation de Connaissances (Teacher ➔ Student)

> **Moteur :** LAAFI_AI_IVA Engine v2.0  
> **Objectif :** Transférer l'expertise diagnostique de `ConvNeXt-Base` (Teacher, 350 MB) vers `MobileNetV4-Small` (Student, 15 MB FP32 / 3.8 MB INT8) sans perte de sensibilité clinique.

---

## 1. Schéma de Distillation Hybride

```text
               ┌─────────────────────────────────────┐
               │ Image d'Entrée JSC (384 x 384 x 3)  │
               └─────────────────────────────────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
   ┌───────────────────────────┐     ┌───────────────────────────┐
   │ Teacher : ConvNeXt-Base   │     │ Student : MobileNetV4     │
   │ (Poids R&D Gelés)         │     │ (Entraînable avec LLRD)   │
   └───────────────────────────┘     └───────────────────────────┘
                 │                                 │
     Logits T    ▼ Attention Map       Logits S    ▼ Attention Map
     (T = 4.0)   │ (Feature Hook)      (T = 4.0)   │ (Feature Hook)
                 └──────────────┐ ┌────────────────┘
                                ▼ ▼
                 ┌───────────────────────────────┐
                 │ BinaryHybridKDLoss            │
                 │ • Soft-BCE (Distillation)     │
                 │ • Hard Asymmetric Focal Loss  │
                 │ • Spatial Attention Transfer  │
                 └───────────────────────────────┘
```

---

## 2. Formulation Mathématique de la Perte

$$\mathcal{L}_{\text{total}} = (1 - \alpha_{\text{kd}}) \cdot \mathcal{L}_{\text{AsymFocal}}(y, \hat{y}_{\text{student}}) + \alpha_{\text{kd}} \cdot T^2 \cdot \mathcal{L}_{\text{SoftBCE}}\left(\sigma\left(\frac{z_{\text{student}}}{T}\right), \sigma\left(\frac{z_{\text{teacher}}}{T}\right)\right) + \beta_{\text{attn}} \cdot \mathcal{L}_{\text{AttnTransfer}}$$

### Paramètres Retenus :
- Température de distillation : $T = 4.0$
- Poids de distillation : $\alpha_{\text{kd}} = 0.60$
- Poids d'attention spatiale : $\beta_{\text{attn}} = 0.10$
- Optimiseur : AdamW avec `CosineAnnealingLR` et LLRD ($LR_{\text{backbone}} = 10^{-5}, LR_{\text{head}} = 10^{-4}$).
