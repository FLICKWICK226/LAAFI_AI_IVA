---
name: kaggle-execution
description: >
  Autonomous Kaggle CI/CD ML execution engine. Automates authentication verification,
  kernel metadata management, pushing notebooks for remote GPU training (Tesla T4/P100),
  asynchronous run monitoring, and downloading trained models (.pt, .onnx) and clinical reports.
---

# Kaggle Execution Engine (CI/CD ML)

Ce skill standardise et automatise l'exécution distante des pipelines d'entraînement de Machine Learning sur les serveurs et accélérateurs GPU Kaggle sans intervention manuelle sur le navigateur.

---

## 🛠️ Commandes du Script Helper

Le script helper [`scripts/kaggle_runner.py`](file:///c:/Users/Rodolpho%20Gouba/Music/LAAFI_AI_IVA/.agents/skills/kaggle-execution/scripts/kaggle_runner.py) expose les sous-commandes suivantes :

```bash
# 1. Vérification de l'authentification API
python .agents/skills/kaggle-execution/scripts/kaggle_runner.py check-auth

# 2. Liste des notebooks distants de l'utilisateur
python .agents/skills/kaggle-execution/scripts/kaggle_runner.py list-kernels

# 3. Pousser et déclencher l'entraînement distant sur GPU
python .agents/skills/kaggle-execution/scripts/kaggle_runner.py push --kernel-dir ./notebooks

# 4. Interroger le statut d'un run (QUEUED, RUNNING, COMPLETE, ERROR)
python .agents/skills/kaggle-execution/scripts/kaggle_runner.py status --kernel-id flickwick/laafi-ai-via

# 5. Télécharger les artefacts générés (poids .pt, .onnx, CSV, figures)
python .agents/skills/kaggle-execution/scripts/kaggle_runner.py pull-outputs --kernel-id flickwick/laafi-ai-via --output-dir ./outputs/kaggle_remote

# 6. Extraire les figures PNG base64 d'un notebook exécuté
python .agents/skills/kaggle-execution/scripts/kaggle_runner.py extract-figures --notebook-path ./outputs/pulled_notebooks/v1/laafi-ai-via.ipynb --output-dir ./outputs/figures
```

---

## 📋 Protocole d'Exécution en 5 Étapes

### Étape 1 : Pre-Flight Check (Authentification)
Vérifier que les identifiants `~/.kaggle/kaggle.json` sont actifs :
```bash
python .agents/skills/kaggle-execution/scripts/kaggle_runner.py check-auth
```

### Étape 2 : Préparation de `kernel-metadata.json`
S'assurer que le fichier `kernel-metadata.json` présent dans le dossier du notebook cible (`notebooks/`) contient :
- L'identifiant complet : `"id": "<username>/<slug>"`
- Le chemin du notebook : `"code_file": "<nom_notebook>.ipynb"`
- L'accélération matérielle : `"enable_gpu": "true"`
- L'accès réseau : `"enable_internet": "true"`
- Les sources de données (datasets / compétitions).

### Étape 3 : Déclenchement Distant (Push)
Pousser le code et lancer l'exécution :
```bash
python .agents/skills/kaggle-execution/scripts/kaggle_runner.py push --kernel-dir ./notebooks
```

### Étape 4 : Surveillance Asynchrone (Non-Bloquante)
Interroger périodiquement le statut :
```bash
python .agents/skills/kaggle-execution/scripts/kaggle_runner.py status --kernel-id <username>/<slug>
```

### Étape 5 : Rapatriement & Extraction des Résultats
Dès que le statut est `COMPLETE` :
1. Télécharger les fichiers sous `/kaggle/working/` :
   ```bash
   python .agents/skills/kaggle-execution/scripts/kaggle_runner.py pull-outputs --kernel-id <username>/<slug> --output-dir ./outputs
   ```
2. Si le notebook contient des figures intégrées (Matplotlib/Seaborn/Grad-CAM) :
   ```bash
   python .agents/skills/kaggle-execution/scripts/kaggle_runner.py extract-figures --notebook-path ./outputs/pulled_notebooks/<slug>.ipynb --output-dir ./outputs/figures
   ```
