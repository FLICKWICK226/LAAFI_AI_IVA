# 📋 TABLEAU KANBAN OFFICIEL : LAAFI_AI_IVA ENGINE
## Cadrage : Assistant d'Aide au Triage & Éligibilité Anatomique (Dataset Intel-MobileODT)
**Responsable du Projet :** Rodolpho Gouba  
**Superviseur ML :** Senior Computer Vision & Medical AI Expert  
**Dernière mise à jour :** Août 2026  

---

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   VUE D'ENSEMBLE DU FLUX                               │
│                                                                                        │
│  [ 📋 BACKLOG ] ──► [ 📌 TO DO ] ──► [ ⚙️ IN PROGRESS ] ──► [ 🧪 TEST ] ──► [ ✅ DONE ] │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 EPIC 1 : Assainissement Data & Quarantaine Intel-MobileODT (Semaine 1)

| ID | Tâche / Ticket | Priorité | Statut | Description Technique & Critères d'Acceptation |
| :--- | :--- | :---: | :---: | :--- |
| **DATA-01** | **Audit d'intégrité & Purge des corruptions JPEG** | `P0 (Bloquant)` | ✅ DONE | Implémenté via `src/data/quality_filter.py` et `src/data/verify_dataset.py` avec isolation des headers corrompus. |
| **DATA-02** | **Dédoublonnage & Clustering Perceptuel (dHash/aHash)** | `P0` | ✅ DONE | `cluster_images_by_perceptual_hash` sans filtre de label (Hamming $\le 6$) avec isolation dans `reports/ambiguous_clusters.csv`. |
| **DATA-03** | **Module de Filtrage Qualité Automatique (Non-Clinique)** | `P1` | ✅ DONE | `CervicalImageQualityFilter` opérationnel (Variance Laplacienne, saturation flash, sous-exposition). |
| **DATA-04** | **Génération du Rapport de Rejet Technique (`quality_report.csv`)** | `P2` | ✅ DONE | Méthode `audit_dataset_manifest` générant pour chaque image : résolution, netteté, saturation, statut (Accepté / Rejeté + motif). |

---

## 🗂️ EPIC 2 : Séparation des Manifestes & Split Anti-Fuite Patiente (Semaine 2)

| ID | Tâche / Ticket | Priorité | Statut | Description Technique & Critères d'Acceptation |
| :--- | :--- | :---: | :---: | :--- |
| **DATA-05** | **Suppression de l'Hérésie `targets > 0` dans tout le code** | `P0 (Critique)` | ✅ DONE | Remplacement par le vrai mapping anatomique tri-classes `Type_1` (0), `Type_2` (1), `Type_3` (2). |
| **DATA-06** | **Création du `manifest_anatomy.csv`** | `P1` | ✅ DONE | `generate_patient_clusters_and_splits` exporte `manifest_anatomy.csv` reliant chaque cliché à sa classe et son split. |
| **DATA-07** | **Partitionnement `StratifiedGroupKFold` par Patiente** | `P0 (Sécurité)` | ✅ DONE | Split Train (70%) / Val (15%) / Test (15%) avec garantie mathématique d'absence de fuite patiente (`tests/test_biomedical_integrity.py`). |

---

## ⚡ EPIC 3 : Modèle Baseline Épuré & Fast Training (Semaine 3)

| ID | Tâche / Ticket | Priorité | Statut | Description Technique & Critères d'Acceptation |
| :--- | :--- | :---: | :---: | :--- |
| **MOD-01** | **Intégration d'un Backbone Léger ($224 \times 224$)** | `P1` | ✅ DONE | Résolution 224x224, batch size 32, support `convnext_small` / `mobilenetv4` (5x plus rapide sur T4). |
| **MOD-02** | **Differential Learning Rate ($10^{-3}$ Tête / $10^{-4}$ Backbone)** | `P1` | ✅ DONE | Permet à la nouvelle tête dense de s'aligner rapidement sans étouffer les gradients. |
| **MOD-03** | **Loss Multi-Classes Tri-Classes (`CrossEntropyLoss`)** | `P1` | ✅ DONE | `calculate_anatomical_metrics` + `CrossEntropyLoss` multi-classes 3 sorties intégrés. |
| **MOD-04** | **Précision Mixte AMP & Scheduler Cosine Annealing** | `P1` | ✅ DONE | `torch.amp.GradScaler` et `CosineAnnealingLR` configurés et poussés sur GitHub. |

---

## 📊 EPIC 4 : Évaluation & Triage Clinique (Semaine 4)

| ID | Tâche / Ticket | Priorité | Statut | Description Technique & Critères d'Acceptation |
| :--- | :--- | :---: | :---: | :--- |
| **EVAL-01** | **Matrice de Confusion 3x3 Réelle (Type 1 vs 2 vs 3)** | `P1` | ✅ DONE | Évaluation à l'aveugle sur le Test Set avec métriques macro-F1, précision et rappel par classe ([`src/utils/metrics.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/src/utils/metrics.py)). |
| **EVAL-02** | **Aide au Triage & Recommandation Post-Acquisition** | `P1` | ✅ DONE | Moteur de triage SaMD opérationnel (`calculate_clinical_triage_metrics`) : Éligible Traitement vs Référer CHU. |
| **EVAL-03** | **Export ONNX & Benchmark de Latence Mobile** | `P1` | ✅ DONE | Export du modèle unifié en ONNX Opset 14 (`src/models/export_onnx.py`) validé par `onnx.checker` ([`tests/test_models_forward.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/tests/test_models_forward.py)). |
| **EVAL-04** | **Documentation SaMD & Traçabilité MLOps** | `P1` | ✅ DONE | Feuilles de route et journaux de bord complétés ([`docs/sprint_roadmap.md`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/docs/sprint_roadmap.md), [`reports/agent_log.md`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/reports/agent_log.md)). |

---

## 🎯 Statut Global du Projet (Sprints 1, 2, 3 Complétés) :
✅ **Tous les jalons (J1, J2, J3) sont atteints et validés par 32 tests automatisés.** Le pipeline est entièrement prêt pour le banc de test distant et l'audit SaMD.


