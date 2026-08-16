# 📋 TASK_TRACKER_WEEK2.md — Suivi d'Exécution Semaine 2 (Distillation Hybride)

> **Projet :** LAAFI_AI_IVA  
> **Objectif Semaine 2 :** Transférer les connaissances de `ConvNeXt-Base` (Teacher) vers `MobileNetV4-Small` (Student, $384 \times 384$) via Distillation Hybride (Soft-BCE + Spatial Attention Transfer).

---

## 🚦 Tableau de Suivi des Tâches

| ID | Statut | Tâche | Description / Livrable |
| :--- | :---: | :--- | :--- |
| **S1-FIX** | ✅ Completed | **Nettoyage & Integrity Check Dataset** | Créer `src/data/verify_dataset.py` pour purger les JPEGs corrompus. |
| **S2-J08** | ✅ Completed | **Gel & Inspection du Teacher** | Configuration gelée de `ConvNeXt-Base` (`requires_grad=False`). |
| **S2-J09** | ✅ Completed | **Implémentation KD Loss** | Créé `src/distillation/kd_loss.py` (`BinaryHybridKDLoss` + Attn Transfer). |
| **S2-J10** | ✅ Completed | **Architecture Student** | Créé `src/models/student_model.py` (`MobileNetV4-Small` $384 \times 384$ + Hooks Attn). |
| **S2-J11** | ✅ Completed | **Script d'Entraînement Distillation** | Créé `experiments/run_distillation.py` avec `CosineAnnealingLR` & LLRD. |
| **S2-J12** | 🚀 Ready | **Exécution Distillation (15 Époques)** | Prêt à exécuter sur GPU (Kaggle/Colab ou local). |
| **S2-J13** | ⏳ Pending | **Évaluation FP32 & Gate Check** | Vérifier Sensibilité $\ge 95\%$ et Spécificité $\ge 85\%$ sur `test.csv`. |

---

## 📊 Registre des Expérimentations

| Modèle | Loss | Res | AUC Val | Spec @ Rec>=95% | Taille FP32 | Statut |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **ConvNeXt-Base (Teacher)** | Asymmetric | $384 \times 384$ | 0.9130 | 68.6% | ~350 MB | ✅ Baseline |
| **MobileNetV4-Small (Student)** | KD Hybride | $384 \times 384$ | - | - | - | ⏳ En cours |
