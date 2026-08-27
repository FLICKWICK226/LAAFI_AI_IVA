# ⏱️ Audit de Performance MLOps : Analyse du Temps par Epoch (Kaggle GPU)

**Projet :** LAAFI_AI_IVA (CADx SaMD - ConvNeXt-Small / PyTorch / Albumentations)  
**Contexte :** Ralentissement critique d'entraînement (~1994 s/epoch, soit ~33 min/epoch pour 20 epochs = >10 h).  
**Objectif :** Identifier les causes racines du goulot d'étranglement sans modifier les hyperparamètres cliniques.

---

## 1. 📊 Résumé Exécutif

L'entraînement affiche un temps anormal de **~33 min/epoch** (1994 s), soit un ralentissement d'un facteur **10× à 15×** par rapport au débit nominal attendu sur Nvidia T4 pour un ConvNeXt-Small en 224×224 (~1,5 à 3 min/epoch). La cause racine dominante est une **famine GPU induite par le CPU et les I/O disque** : lecture synchrone d'images JPEG non compressées/lourdes sur disque virtuel Kaggle combinée à des transformations CPU unitaires bloquantes (`cv2.imread`, décodage et augmentations) dans un DataLoader sous-dimensionné.

---

## 2. 🔍 Hypothèses Classées par Ordre de Vraisemblance

### 1. Goulot d'étranglement I/O Disque et Décodage Image dans le DataLoader (GPU Starvation)
* **Indices POUR :** Le chargement direct par chemin de fichier à chaque `__getitem__` sur le système de fichiers `/kaggle/input/` (disque virtuel partagé/FUSE lent) sature l'I/O. Le décodage JPEG d'images haute résolution d'origine via OpenCV bloque chaque worker.
* **Indices CONTRE :** Aucun à ce stade.
* **Données manquantes pour trancher :** Ratio d'utilisation GPU (`GPU-Util %` vs `Volatile GPU-Util`) et métrique d'attente batch (`data_time` vs `compute_time`).
* **Gain de temps potentiel :** **70 % à 85 % de réduction du temps par epoch** (passage de 33 min à ~3-5 min).

### 2. Surcharge CPU liée aux Transformations & Augmentations Albumentations / Bruit procédural
* **Indices POUR :** Même redimensionnées en amont, les opérations CPU séquentielles (`A.Defocus`, `A.MotionBlur`, `A.RandomSunFlare`, `A.ColorJitter` + masquage Perlin et manipulation de tableaux NumPy par échantillon) consomment des dizaines de millisecondes par image. Sur 2 vCPU Kaggle, cela sature le thread de préparation.
* **Indices CONTRE :** Les tableaux sont petits (224×224), ce qui limite l'explosion mémoire mais n'empêche pas la latence CPU cumulative.
* **Données manquantes pour trancher :** Profilage unitaire temps/fonction (`time.perf_counter()` autour de `transform()` vs `add_blood_or_mucus()` vs `cv2.imread`).
* **Gain de temps potentiel :** **30 % à 50 % de gain**.

### 3. Sous-utilisation du parallélisme DataLoader (`num_workers`, `pin_memory`, `persistent_workers`)
* **Indices POUR :** Sur Kaggle (2 vCPUs alloués par défaut avec GPU T4), un `num_workers=0` (exécution sur le thread principal) ou `num_workers=2` sans `persistent_workers=True` réinstancie les processus à chaque epoch et détruit le cache RAM des masques Perlin.
* **Indices CONTRE :** La présence de `pin_memory=True` (si actif) limite le coût de transfert Host-to-Device.
* **Données manquantes pour trancher :** Valeurs exactes passées aux arguments du DataLoader au moment du run.
* **Gain de temps potentiel :** **25 % à 40 % de gain**.

### 4. Fallback silencieux ou surcharge du compilateur `torch.compile`
* **Indices POUR :** Si `torch.compile(model)` est activé sur un backend non compatible ou déclenche des recompilations continues (dynamisme de shape / graph breaks), l'overhead de compilation s'ajoute au début de chaque phase.
* **Indices CONTRE :** Le premier epoch dure 33 min ; une recompilation `Inductor` prend 1 à 3 min au premier batch, pas 33 min en continu.
* **Données manquantes pour trancher :** Logs détaillés `TORCH_LOGS="recompiles,graph_breaks"`.
* **Gain de temps potentiel :** **5 % à 15 % de gain** (ou neutralisation des freezes).

### 5. Exécution partielle ou totale sur CPU (CUDA non engagé)
* **Indices POUR :** Un modèle ConvNeXt-Small exécuté entièrement sur CPU met exactement ~30-40 min par epoch sur ~1500 images.
* **Indices CONTRE :** Non tranchable avec les logs actuels (dépend de la validation de la commande `nvidia-smi` dans le conteneur).
* **Données manquantes pour trancher :** Log de `torch.cuda.is_available()`, `device` effectif du modèle et tenseurs images.
* **Gain de temps potentiel :** **90 % de réduction** si le job tournait en réalité sur CPU.

---

## 3. 📋 Checklist de Diagnostic (À exécuter sur le Notebook)

| Étape | Commande / Code à insérer dans une cellule de test | Ce que l'observation confirme ou infirme |
| :--- | :--- | :--- |
| **D1. Charge & Mémoire GPU** | `!nvidia-smi -l 1` (ou appel dans une cellule de monitoring) | Vérifie si `GPU-Util` est à **~100 %** (GPU Compute Bound) ou oscille à **< 10 %** (CPU/IO Bound = Famine). |
| **D2. Profilage Débit DataLoader (Pur I/O)** | ```python<br>import time<br>t0 = time.perf_counter()<br>for i, (imgs, targets, _) in enumerate(train_loader):<br>    if i >= 50: break<br>print(f"Débit DataLoader pur : {50 / (time.perf_counter() - t0):.2f} batch/s")<br>``` | Isole la vitesse d'alimentation sans passe forward/backward. Si < 5 batch/s, le problème est 100 % dans le DataLoader. |
| **D3. Profilage Granulaire `__getitem__`** | ```python<br>import time<br># Mesurer 100 itérations unitaires de __getitem__<br>times = {'io': [], 'perlin': [], 'alb': []}<br>for idx in range(100):<br>    # Mesure cv2.imread + resize vs perlin vs transform<br>    ...<br>``` | Identifie la fonction exacte responsable de la latence (I/O disque vs Albumentations vs Perlin). |
| **D4. Validation PyTorch Profiler** | ```python<br>with torch.profiler.profile(<br>    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],<br>    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),<br>    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_trace')<br>) as prof:<br>    for i, batch in enumerate(train_loader):<br>        # step<br>        prof.step()<br>``` | Fournit la répartition exacte entre `DataLoader overhead`, `cudaMemcpyAsync`, `Forward` et `Backward`. |
| **D5. Vérification Allocation Workers** | `print(os.cpu_count(), train_loader.num_workers, train_loader.pin_memory, train_loader.persistent_workers)` | Confirme que le pipeline n'utilise pas `num_workers=0` et tire parti du multithreading. |

---

## 4. 💡 Recommandations d'Optimisation

### A. Pipeline de Données (Priorité Critique)
* **Mise en cache RAM / Shm (Disque RAM) :** Si la taille du dataset pré-redimensionné (224×224) représente moins de 1 à 2 Go, pré-charger l'ensemble des tenseurs ou tableaux NumPy en RAM au moment de l'initialisation du Dataset pour ramener l'I/O disque à **0 ms**.
* **Configuration DataLoader :**
  - Régler `num_workers=2` (maximum recommandé sur VM Kaggle 2 vCPUs) combiné impérativement avec `persistent_workers=True` pour éviter la réinitialisation des pools de processus à chaque epoch.
  - S'assurer que `pin_memory=True` est actif pour paralléliser les transferts CPU $\to$ VRAM via DMA.
* **Audit des Augmentations CPU-intensives :** Les filtres comme `A.RandomSunFlare` et `A.Defocus` sont réputés lents sous CPU. Comparer le débit d'un epoch en désactivant temporairement les filtres optiques complexes vs augmentations géométriques pures (Flip, Rotate).

### B. Architecture & Moteur d'Exécution
* **Précision Mixte (AMP) :** Confirmer que `torch.amp.autocast('cuda')` et `GradScaler` englobent bien le forward/loss pour exploiter les Tensor Cores du GPU T4.
* **Format Mémoire Contigu :** Utiliser le format `memory_format=torch.channels_last` pour ConvNeXt, particulièrement optimisé pour les architectures convolutionnelles sur GPU Nvidia.
* **Vérification `torch.compile` :** Si activé, tester un run standard sans compilation (`model = raw_model`) pour s'assurer que `torch.compile` n'induit pas de surcoût sur des graphes dynamiques.
