# 📋 TASK_TRACKER_WEEK3.md — Suivi d'Exécution Semaine 3 (Quantification INT8 & ExecuTorch)

> **Projet :** LAAFI_AI_IVA  
> **Objectif Semaine 3 :** Compresser le modèle Student `MobileNetV4-Small` ($384 \times 384$) sous $15\text{ MB}$ en INT8 avec une perte de précision clinique $< 0.5\%$.

---

## 🚦 Tableau de Suivi des Tâches

| ID | Statut | Tâche | Description / Livrable |
| :--- | :---: | :--- | :--- |
| **S3-J15** | ✅ Completed | **Export Graphe PyTorch 2.x** | Créé `export/export_executorch.py` (`torch.export` & ONNX). |
| **S3-J17** | ✅ Completed | **Post-Training Quantization (PTQ)** | Créé `export/ptq_quantizer.py` (Calibration 500 images). |
| **S3-J19** | ✅ Completed | **Quantization Gate Check** | Créé `export/evaluate_quantized.py` (Perte métrique $< 0.5\%$). |
| **S3-J20** | ✅ Completed | **Fallback QAT** | Créé `export/qat_trainer.py` (Ré-entraînement 3-5 époques fake-quant). |

---

## 📊 Registre des Tailles et Métriques de Quantification

| Modèle | Format | Taille Disque | AUC Test | Spec @ Rec>=95% | Delta FP32 | Statut Gate Check |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Student MobileNetV4** | FP32 | ~15 MB | - | - | Baseline | - |
| **Student MobileNetV4** | INT8 PTQ | ~3.8 MB | - | - | - | ⏳ À évaluer |
| **Student MobileNetV4** | INT8 QAT | ~3.8 MB | - | - | - | ⏳ Fallback si dégradation |
