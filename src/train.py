import os
import sys
import yaml
import shutil
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Anti-fragmentation mémoire VRAM CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import importlib
from src.utils.seed import seed_everything
import src.data.augmentations
import src.data.dataset
importlib.reload(src.data.augmentations)
importlib.reload(src.data.dataset)

from src.data.dataset import IVADataset
from src.models.classifier_lesion import IVALesionClassifierStage2
from src.utils.metrics import (
    calculate_clinical_metrics,
    evaluate_threshold_grid,
    calculate_anatomical_metrics,
    calculate_clinical_triage_metrics
)

def fast_sync_to_ssd(src_dir: str, dst_dir: str) -> None:
    """
    Copieur de données intelligent avec barre de progression interactive tqdm.
    Évite le gel synchrone sur Google Drive FUSE.
    """
    if not os.path.exists(src_dir):
        return

    if os.path.exists(dst_dir):
        src_files_count = sum([len(files) for _, _, files in os.walk(src_dir)])
        dst_files_count = sum([len(files) for _, _, files in os.walk(dst_dir)])
        if dst_files_count >= src_files_count and dst_files_count > 0:
            print(f"⚡ Données déjà synchronisées sur le SSD local ({dst_files_count} fichiers présent(s)). Copie ignorée.")
            return

    os.makedirs(dst_dir, exist_ok=True)
    
    all_files_to_copy = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            src_file = os.path.join(root, f)
            rel_path = os.path.relpath(src_file, src_dir)
            dst_file = os.path.join(dst_dir, rel_path)
            if not os.path.exists(dst_file):
                all_files_to_copy.append((src_file, dst_file))

    if not all_files_to_copy:
        print("⚡ Tous les fichiers sont déjà à jour sur le SSD local.")
        return

    print(f"📦 Synchronisation de {len(all_files_to_copy)} fichiers du Drive vers le SSD Local Colab...")
    for src_file, dst_file in tqdm(all_files_to_copy, desc="Copie SSD Local", unit="fichier"):
        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
        shutil.copy2(src_file, dst_file)

    print("✅ Copie sur le SSD local terminée avec succès !")

def train_laafi_ai_model(config_path: str = "./config/config.yaml") -> None:
    """
    Moteur d'entraînement unifié (Single-Head 3-Classes) pour Stage 2 (CADx).
    """
    print("\n" + "="*70)
    print("🚀 STEP [1/6] : Chargement de la configuration & Graine aléatoire...")
    print("="*70)

    # Résolution robuste multi-chemins du fichier config.yaml
    if not os.path.exists(config_path):
        candidates = [
            os.path.join(os.getcwd(), config_path),
            os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"),
            os.path.join(os.path.dirname(__file__), "..", config_path),
            "/kaggle/working/LAAFI_AI_IVA/config/config.yaml",
            "/kaggle/working/config/config.yaml",
            "./config/config.yaml",
            "../config/config.yaml",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                config_path = cand
                break

    print(f"📖 Lecture de la configuration depuis : {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['project']['seed'])
    
    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            # Test d'intégrité CUDA (vérifie la compatibilité de l'architecture GPU avec PyTorch)
            _t = torch.zeros(1, device='cuda') + 1.0
            device = torch.device("cuda")
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"⚠️ GPU CUDA détecté mais incompatible avec cette version PyTorch ({e}).")
            print("🔄 Basculement automatique sécurisé sur CPU.")
            device = torch.device("cpu")

    print(f"🖥️ Exécution de l'entraînement sur : {device}")

    print("\n" + "="*70)
    print("🚀 STEP [2/6] : Synchronisation et résolution des chemins de données...")
    print("="*70)

    if 'google.colab' in sys.modules or os.path.exists("/content/drive/MyDrive/LAAFI_AI_IVA/data"):
        drive_data_dir = "/content/drive/MyDrive/LAAFI_AI_IVA/data"
        fast_data_dir = "/content/data_fast"
        fast_sync_to_ssd(drive_data_dir, fast_data_dir)

    processed_dir = cfg['paths']['processed_data_dir']
    masks_dir = cfg['paths']['synthetic_masks_dir']
    checkpoints_dir = cfg['paths']['checkpoints_dir']
    logs_dir = cfg['paths']['logs_dir']
    reports_dir = cfg['paths']['reports_dir']
    figures_dir = cfg['paths']['figures_dir']

    if os.path.exists("/kaggle/working"):
        if os.path.exists("/kaggle/working/data/processed/train.csv"):
            processed_dir = "/kaggle/working/data/processed"
        elif os.path.exists("/kaggle/working/LAAFI_AI_IVA/data/processed/train.csv"):
            processed_dir = "/kaggle/working/LAAFI_AI_IVA/data/processed"
        elif not os.path.exists(os.path.join(processed_dir, "train.csv")) and os.path.exists("../data/processed/train.csv"):
            processed_dir = "../data/processed"
            
        if os.path.exists("/kaggle/working/data/synthetic_masks"):
            masks_dir = "/kaggle/working/data/synthetic_masks"
        elif os.path.exists("/kaggle/working/LAAFI_AI_IVA/data/synthetic_masks"):
            masks_dir = "/kaggle/working/LAAFI_AI_IVA/data/synthetic_masks"
            
        checkpoints_dir = "/kaggle/working/models/checkpoints"
        logs_dir = "/kaggle/working/outputs/logs"
        reports_dir = "/kaggle/working/outputs/reports"
        figures_dir = "/kaggle/working/outputs/figures"

    num_workers = min(2, os.cpu_count() or 2) if device.type == 'cuda' else 0
    pin_memory = True if device.type == 'cuda' else False

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("\n" + "="*70)
    print("🚀 STEP [3/6] : Chargement des Datasets PyTorch (Train & Val)...")
    print("="*70)

    train_csv_path = os.path.join(processed_dir, "train.csv")
    val_csv_path = os.path.join(processed_dir, "val.csv")

    print(f"🔍 Chargement train.csv depuis : {train_csv_path}")
    print(f"🔍 Chargement val.csv depuis   : {val_csv_path}")

    train_dataset = IVADataset(
        csv_file=train_csv_path,
        is_train=True,
        masks_dir=masks_dir,
        perlin_proba=cfg['augmentations']['perlin_noise_proba']
    )
    val_dataset = IVADataset(
        csv_file=val_csv_path,
        is_train=False
    )

    batch_size = cfg['stage2_classifier']['batch_size']
    accum_steps = cfg['stage2_classifier'].get('gradient_accumulation_steps', 1)
    use_weighted_sampler = cfg['stage2_classifier'].get('use_weighted_sampler', False)

    # Échantillonnage naturel (ou pondéré doux si explicitement activé)
    if use_weighted_sampler and hasattr(train_dataset, 'df') and 'target' in train_dataset.df.columns and len(train_dataset.df) > 0:
        train_labels = train_dataset.df['target'].values
        class_counts = np.bincount(train_labels, minlength=3)
        total_samples = len(train_labels)
        class_weights = total_samples / (3.0 * np.maximum(class_counts, 1).astype(float))
        sample_weights = [class_weights[int(t)] for t in train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    print(f"📊 Dataset chargé avec succès : {len(train_dataset)} train, {len(val_dataset)} val.")

    if len(train_dataset) == 0:
        raise ValueError(f"❌ Le dataset d'entraînement est vide (0 échantillon dans {train_csv_path}). Veuillez ré-exécuter la Cellule 4 (generate_patient_clusters_and_splits).")

    print("\n" + "="*70)
    print(f"🚀 STEP [4/6] : Initialisation du Modèle Unifié ({cfg['stage2_classifier']['backbone']})...")
    print("="*70)

    raw_model = IVALesionClassifierStage2(
        backbone_name=cfg['stage2_classifier']['backbone'],
        pretrained=True,
        num_classes=int(cfg['stage2_classifier'].get('num_classes', 3)),
        drop_rate=float(cfg['stage2_classifier'].get('drop_rate', 0.2))
    ).to(device)

    if device.type == "cuda":
        raw_model = raw_model.to(memory_format=torch.channels_last)

    model = raw_model

    # Fonction de coût unifiée et symétrique avec label smoothing
    label_smoothing = float(cfg['stage2_classifier'].get('label_smoothing', 0.05))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Optimizer avec Differential Learning Rate (Backbone à 1e-4, Tête à 1e-3)
    backbone_lr = float(cfg['stage2_classifier'].get('learning_rate', 1e-4))
    head_lr = float(cfg['stage2_classifier'].get('head_learning_rate', 1e-3))
    weight_decay = float(cfg['stage2_classifier'].get('weight_decay', 1e-4))

    optimizer = torch.optim.AdamW([
        {'params': raw_model.backbone.parameters(), 'lr': backbone_lr},
        {'params': raw_model.head.parameters(), 'lr': head_lr}
    ], weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['stage2_classifier']['epochs']
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_val_loss = float('inf')
    best_val_auc = 0.0
    start_epoch = 1
    no_improve_epochs = 0

    warmup_epochs = cfg['stage2_classifier'].get('warmup_epochs', 2)
    patience = cfg['stage2_classifier'].get('early_stopping_patience', 5)

    print("\n" + "="*70)
    print("🚀 STEP [5/6] : Vérification du Checkpoint de reprise...")
    print("="*70)

    latest_checkpoint_path = os.path.join(checkpoints_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_checkpoint_path):
        print(f"🔄 Checkpoint de reprise trouvé : {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            raw_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and scaler is not None and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_auc = checkpoint.get('best_val_auc', 0.0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
            print(f"✅ Reprise réussie à l'epoch {start_epoch} (Meilleur Val AUC : {best_val_auc:.4f}, Val Loss : {best_val_loss:.4f})")
        except Exception as e_res:
            print(f"⚠️ Checkpoint incompatible ou corrompu ({e_res}). Démarrage d'un nouvel entraînement.")

    if start_epoch > cfg['stage2_classifier']['epochs']:
        print(f"🎉 Entraînement déjà achevé ({cfg['stage2_classifier']['epochs']} epochs).")
        return

    print("\n" + "="*70)
    print(f"🚀 STEP [6/6] : Démarrage de la boucle d'entraînement (Epoch {start_epoch} à {cfg['stage2_classifier']['epochs']})...")
    print("="*70)

    live_status_path = os.path.join(logs_dir, "live_status.json")

    for epoch in range(start_epoch, cfg['stage2_classifier']['epochs'] + 1):
        
        # Warmup Backbone Freeze
        if epoch <= warmup_epochs:
            print(f"🔒 Epoch {epoch}/{warmup_epochs} - Backbone gelé (Warmup tête de classification).")
            for param in raw_model.backbone.parameters():
                param.requires_grad = False
        else:
            if epoch == warmup_epochs + 1:
                print("🔓 Phase de warmup terminée ! Déblocage des poids du backbone ConvNeXt.")
            for param in raw_model.backbone.parameters():
                param.requires_grad = True

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['stage2_classifier']['epochs']}", unit="batch")
        for step, (images, targets, _) in enumerate(pbar):
            if device.type == 'cuda':
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            else:
                images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(images)
                loss = criterion(logits, targets)
                total_loss = loss / accum_steps

            scaler.scale(total_loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += total_loss.item() * accum_steps
            pbar.set_postfix({"Loss": f"{total_loss.item() * accum_steps:.4f}"})

            if step % 10 == 0 or step == len(train_loader) - 1:
                live_info = {
                    "status": "training",
                    "epoch": epoch,
                    "max_epochs": cfg['stage2_classifier']['epochs'],
                    "batch": step + 1,
                    "total_batches": len(train_loader),
                    "current_loss": float(total_loss.item() * accum_steps)
                }
                with open(live_status_path, "w", encoding="utf-8") as f_live:
                    json.dump(live_info, f_live, indent=4)

        scheduler.step()
        train_loss /= len(train_loader) if len(train_loader) > 0 else 1

        # Phase de Validation
        model.eval()
        val_loss = 0.0
        val_targets, val_probs = [], []
        val_raw_targets, val_all_probs = [], []
        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc=f"Validation Epoch {epoch}", leave=False):
                if device.type == 'cuda':
                    images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
                else:
                    images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(images)
                    loss_v = criterion(logits, targets)
                    probs = torch.softmax(logits, dim=1) # [B, 3]

                val_loss += loss_v.item() * len(targets)
                # Probabilité clinique de positivité (Type 2 Lésionnel + Type 3 Invalide/Suspect = 1 - Type 1)
                prob_positive = (probs[:, 1] + probs[:, 2]).cpu().numpy()
                val_targets.extend((targets > 0).cpu().numpy())
                val_probs.extend(prob_positive)
                val_raw_targets.extend(targets.cpu().numpy())
                val_all_probs.extend(probs.cpu().numpy())

        val_loss /= len(val_dataset) if len(val_dataset) > 0 else 1
        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)
        val_raw_targets = np.array(val_raw_targets)
        val_all_probs = np.array(val_all_probs)
        
        best_threshold = 0.50
        best_metrics = {"auc_roc": 0.5, "sensitivity": 0.0, "specificity": 0.0, "f2_score": 0.0}
        anat_metrics = {"macro_f1": 0.0, "macro_auc_roc": 0.5, "accuracy": 0.0}
        triage_metrics = {"safety_specificity_type3": 0.0, "triage_accuracy": 0.0}
        
        if len(val_raw_targets) > 0 and len(np.unique(val_raw_targets)) > 1:
            anat_metrics = calculate_anatomical_metrics(val_raw_targets, val_all_probs)
            triage_metrics = calculate_clinical_triage_metrics(val_raw_targets, val_all_probs)

        if len(val_targets) > 0 and len(np.unique(val_targets)) > 1:
            t_cfg = cfg['stage2_classifier'].get('threshold_calibration', {})
            min_t = float(t_cfg.get('min_t', 0.05))
            max_t = float(t_cfg.get('max_t', 0.95))
            step_t = float(t_cfg.get('step', 0.01))
            target_sens = float(t_cfg.get('target_sensitivity', 0.95))
            min_spec = float(t_cfg.get('min_specificity', 0.50))
            
            grid_res = evaluate_threshold_grid(
                val_targets,
                val_probs,
                min_t=min_t,
                max_t=max_t,
                step=step_t,
                target_sensitivity=target_sens,
                min_specificity=min_spec
            )
            if grid_res['optimal']:
                best_metrics = grid_res['optimal']
                best_threshold = best_metrics['threshold']

        print(f"📊 Epoch {epoch} Terminée | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro-F1: {anat_metrics['macro_f1']:.4f} | Macro-AUC: {anat_metrics['macro_auc_roc']:.4f} | Sensibilité Dépistage: {best_metrics['sensitivity']*100:.1f}% | Spécificité: {best_metrics.get('specificity', 0)*100:.1f}% | Sécurité Type 3: {triage_metrics.get('safety_specificity_type3', 0)*100:.1f}%")

        # Mise à jour des rapports d'entraînement
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        epoch_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auc": float(best_metrics['auc_roc']),
            "val_sensitivity": float(best_metrics['sensitivity']),
            "val_specificity": float(best_metrics.get('specificity', 0.0)),
            "val_f2_score": float(best_metrics.get('f2_score', 0.0)),
            "best_threshold": float(best_threshold)
        }

        csv_report_path = os.path.join(reports_dir, "training_history.csv")
        df_log = pd.DataFrame([epoch_log])
        if not os.path.exists(csv_report_path):
            df_log.to_csv(csv_report_path, index=False)
        else:
            df_log.to_csv(csv_report_path, mode='a', header=False, index=False)

        json_report_path = os.path.join(reports_dir, "training_summary.json")
        history = []
        if os.path.exists(json_report_path):
            try:
                with open(json_report_path, 'r', encoding='utf-8') as f_json:
                    history = json.load(f_json)
            except Exception:
                history = []
        history.append(epoch_log)
        with open(json_report_path, 'w', encoding='utf-8') as f_json:
            json.dump(history, f_json, indent=4)

        # Checkpointing sur Validation AUC et Validation Loss
        is_best = False
        if best_metrics['auc_roc'] > best_val_auc or (abs(best_metrics['auc_roc'] - best_val_auc) < 0.005 and val_loss < best_val_loss):
            is_best = True
            best_val_auc = max(best_val_auc, best_metrics['auc_roc'])
            best_val_loss = min(best_val_loss, val_loss)
            no_improve_epochs = 0
            best_path = os.path.join(checkpoints_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'best_threshold': best_threshold,
                'val_auc': best_val_auc,
                'val_loss': best_val_loss
            }, best_path)
            print(f"💾 Nouveau meilleur modèle (Val AUC: {best_val_auc:.4f}, Val Loss: {best_val_loss:.4f}) -> {best_path} sauvegardé !")
        else:
            no_improve_epochs += 1
            print(f"⏳ Pas d'amélioration depuis {no_improve_epochs}/{patience} epoch(s).")

        latest_dict = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_threshold': best_threshold,
            'best_val_auc': best_val_auc,
            'best_val_loss': best_val_loss,
            'no_improve_epochs': no_improve_epochs
        }
        torch.save(latest_dict, latest_checkpoint_path)

        if no_improve_epochs >= patience:
            print(f"🛑 Arrêt précoce à l'epoch {epoch} (Patience de {patience} epochs atteinte).")
            print(f"🎯 Meilleur Val AUC : {best_val_auc:.4f} | Meilleure Val Loss : {best_val_loss:.4f}")
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cache GPU vidé.")

if __name__ == "__main__":
    train_laafi_ai_model()
