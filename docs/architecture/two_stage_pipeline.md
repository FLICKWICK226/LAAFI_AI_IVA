# 🏗️ Architecture Technique : Pipeline à Deux Étages (CADe / CADx)

> **Moteur :** LAAFI_AI_IVA Engine v2.0  
> **Principe :** Séparation stricte de la localisation anatomique (CADe) et de la classification diagnostique (CADx).

---

## 1. Vue d'Ensemble du Flux de Données

```text
┌─────────────────────────────────┐
│ Image Brute Smartphone (HD)     │
│ Spéculum, Parois, Reflets LED   │
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Stage 1 : CADe (YOLOv8n / v11n) │ ──► Détecte la Jonction Squamo-Columnaire (JSC)
└─────────────────────────────────┘
                 │
                 ▼ Crop ROI (384x384) [Élimination > 70% Arrière-plan parasite]
┌─────────────────────────────────┐
│ Module Reflets-Lite (HSV)       │ ──► Atténuation des reflets spéculaires sans trou noir
└─────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Stage 2 : CADx (Teacher/Student)│ ──► ConvNeXt-Base (R&D) ou MobileNetV4-Small (Edge)
└─────────────────────────────────┘
                 │
                 ▼ Logits [B, 1]
┌─────────────────────────────────┐
│ Calibration Seuil T_opt         │ ──► Sensibilité >= 95.0%
└─────────────────────────────────┘
                 │
                 ▼ Triage Tri-classe
┌─────────────────────────────────┐
│ Vert (<0.20) / Jaune / Rouge    │
└─────────────────────────────────┘
```

---

## 2. Spécifications des Étages

### Étage 1 : CADe (Computer-Aided Detection)
- **Modèle :** `YOLOv8n-det` ou `YOLOv11n-det` ($\sim 3.2\text{ MB}$).
- **Résolution d'Entrée :** $640 \times 640$.
- **Rôle :** Localiser la boîte englobante de la zone cervicale active (col utérin) et produire le rognage centré $384 \times 384$.

### Étage 2 : CADx (Computer-Aided Diagnosis)
- **Modèle Teacher (R&D) :** `ConvNeXt-Base` (88M paramètres, pré-entraîné ImageNet-1k).
- **Modèle Student (Edge) :** `MobileNetV4-Small` ($\sim 3.8\text{ MB}$ INT8).
- **Résolution d'Entrée :** $384 \times 384 \times 3$.
- **Sortie :** Logit binaire calibré pour la détection de lésion acéto-blanche précancéreuse.
