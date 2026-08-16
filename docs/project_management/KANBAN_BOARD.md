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
| **DATA-01** | **Audit d'intégrité & Purge des corruptions JPEG** | `P0 (Bloquant)` | ✅ DONE | Implémenté via `src/data/quality_filter.py` avec isolation des headers corrompus. |
| **DATA-02** | **Dédoublonnage par Hachage SHA-256** | `P1` | 📌 TO DO | Détecter et isoler les images identiques ou quasi-doublons issues du scraping Kaggle. |
| **DATA-03** | **Module de Filtrage Qualité Automatique (Non-Clinique)** | `P1` | ✅ DONE | `CervicalImageQualityFilter` opérationnel (Variance Laplacienne, saturation flash, sous-exposition). |
| **DATA-04** | **Génération du Rapport de Rejet Technique (`quality_report.csv`)** | `P2` | 📌 TO DO | Fichier CSV listant pour chaque image : résolution, score de netteté, saturation, statut (Accepté / Rejeté + motif). |

---

## 🗂️ EPIC 2 : Séparation des Manifestes & Split Anti-Fuite Patiente (Semaine 2)

| ID | Tâche / Ticket | Priorité | Statut | Description Technique & Critères d'Acceptation |
| :--- | :--- | :---: | :---: | :--- |
| **DATA-05** | **Suppression de l'Hérésie `targets > 0` dans tout le code** | `P0 (Critique)` | ✅ DONE | Remplacement par le vrai mapping anatomique tri-classes `Type_1` (0), `Type_2` (1), `Type_3` (2). |
| **DATA-06** | **Création du `manifest_anatomy.csv`** | `P1` | 📌 TO DO | Manifeste propre associant chaque image valide à sa vraie classe anatomique : `Type_1` (0), `Type_2` (1), `Type_3` (2). |
| **DATA-07** | **Partitionnement `StratifiedGroupKFold` par Patiente** | `P0 (Sécurité)` | ✅ DONE | Split Train (70%) / Val (15%) / Test (15%) avec garantie mathématique d'absence de fuite patiente (`test_patient_leakage.py`). |

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
| **EVAL-01** | **Matrice de Confusion 3x3 Réelle (Type 1 vs 2 vs 3)** | `P1` | ✅ DONE | Évaluation à l'aveugle sur le Test Set avec métriques macro-F1, précision et rappel par classe. |
| **EVAL-02** | **Aide au Triage & Recommandation Post-Acquisition** | `P1` | ✅ DONE | Moteur de triage SaMD opérationnel (`calculate_clinical_triage_metrics`) : Éligible Traitement vs Référer CHU. |
| **EVAL-03** | **Export ONNX & Benchmark de Latence Mobile** | `P2` | 📋 BACKLOG | Export du modèle épuré au format ONNX et validation de l'inférence $<50\text{ ms}$ sur CPU mobile. |
| **EVAL-04** | **Documentation SaMD & Model Card Révisée** | `P2` | 📋 BACKLOG | Mise à jour de `MODEL_CARD.md` et `DATA_CARD.md` reflétant le positionnement d'aide au triage anatomique. |

---

## 🎯 Prochaine Action Immédiate (Sprint Actuel) :
👉 **Lancer l'entraînement Weighted CrossEntropy (20 époques) :** Poids de classe inverses automatiques (Type 1 puni 2.4x plus fort) + Moteur de triage SaMD.

