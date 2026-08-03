import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.data.dataset import IVADataset
from src.models.classifier_lesion import IVALesionClassifierStage2
from src.utils.metrics import FocalLoss, calculate_clinical_metrics

def train_laafi_ai_model(config_path: str = "./config/config.yaml") -> None:
    """
    Moteur d'entraînement principal pour Stage 2 (CADx) ultra-optimisé pour GPU Google Colab T4.
    Intègre les Levis de performance 1 (Early Stopping), 2 (torch.compile) et 5 (Warmup Backbone Freeze).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['project']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Exécution de l'entraînement sur : {device}")

    # Mirroring SSD Local Colab (/content/data_fast) AVANT instanciation des Datasets
    if 'google.colab' in sys.modules or os.path.exists("/content/drive/MyDrive/LAAFI_AI_IVA/data"):
        drive_data_dir = "/content/drive/MyDrive/LAAFI_AI_IVA/data"
        fast_data_dir = "/content/data_fast"
        if os.path.exists(drive_data_dir) and not os.path.exists(fast_data_dir):
            print("🚀 Copie unique des données du Google Drive vers le SSD Local Colab (accélération x20)...")
            os.system(f"cp -r {drive_data_dir} {fast_data_dir}")
            print("✅ Copie terminée ! L'entraînement s'exécute désormais sur le SSD local ultra-rapide.")

    num_workers = min(2, os.cpu_count() or 2) if device.type == 'cuda' else 0
    pin_memory = True if device.type == 'cuda' else False

    os.makedirs(cfg['paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(cfg['paths']['logs_dir'], exist_ok=True)

    # Chargement des Datasets
    train_dataset = IVADataset(
        csv_file=os.path.join(cfg['paths']['processed_data_dir'], "train.csv"),
        is_train=True,
        masks_dir=cfg['paths']['synthetic_masks_dir'],
        perlin_proba=cfg['augmentations']['perlin_noise_proba']
    )
    val_dataset = IVADataset(
        csv_file=os.path.join(cfg['paths']['processed_data_dir'], "val.csv"),
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['stage2_classifier']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['stage2_classifier']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    # Initialisation du Modèle Stage 2
    raw_model = IVALesionClassifierStage2(
        backbone_name=cfg['stage2_classifier']['backbone'],
        pretrained=True,
        num_classes_eligibility=3,
        num_classes_pathology=2
    ).to(device)

    # ⚡ LEVIER 2 : Compilateur PyTorch 2.0 (Kernel Fusion) si supporté
    model = raw_model
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(raw_model)
            print("⚡ LEVIER 2 APPLIQUÉ : Modèle compilé via torch.compile() pour fusion de kernels CUDA.")
        except Exception as e_comp:
            print(f"⚠️ Remarque torch.compile non appliqué : {e_comp}")

    # Loss, Optimiseur et Scheduler
    criterion_eligibility = nn.CrossEntropyLoss()
    criterion_pathology = FocalLoss(
        alpha=cfg['stage2_classifier']['focal_loss_alpha'],
        gamma=cfg['stage2_classifier']['focal_loss_gamma']
    )
    
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=float(cfg['stage2_classifier']['learning_rate']),
        weight_decay=float(cfg['stage2_classifier']['weight_decay'])
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['stage2_classifier']['epochs']
    )
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    best_val_auc = 0.0
    start_epoch = 1
    no_improve_epochs = 0

    # Paramètres des leviers 1 et 5
    warmup_epochs = cfg['stage2_classifier'].get('warmup_epochs', 3)
    patience = cfg['stage2_classifier'].get('early_stopping_patience', 5)

    # Système Restart-Safe : Reprise Automatique depuis latest_checkpoint.pt
    latest_checkpoint_path = os.path.join(cfg['paths']['checkpoints_dir'], "latest_checkpoint.pt")
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
            no_improve_epochs = checkpoint.get('no_improve_epochs', 0)
            print(f"✅ Reprise réussie à l'epoch {start_epoch} (Meilleur Val AUC : {best_val_auc:.4f})")
        except Exception as e_res:
            print(f"⚠️ Impossible de charger le checkpoint ({e_res}). Démarrage à zéro.")

    if start_epoch > cfg['stage2_classifier']['epochs']:
        print(f"🎉 Entraînement déjà achevé ({cfg['stage2_classifier']['epochs']} epochs).")
        return

    print(f"🚀 Début de l'entraînement optimisé (Epoch {start_epoch} à {cfg['stage2_classifier']['epochs']})...")
    for epoch in range(start_epoch, cfg['stage2_classifier']['epochs'] + 1):
        
        # 🔒 LEVIER 5 : Gel du backbone pendant les 'warmup_epochs' (Warmup Transfer Learning)
        if epoch <= warmup_epochs:
            print(f"🔒 LEVIER 5 : Epoch {epoch}/{warmup_epochs} - Backbone gelé (Seules les têtes sont entraînées).")
            for param in raw_model.backbone.parameters():
                param.requires_grad = False
        else:
            if epoch == warmup_epochs + 1:
                print(f"🔓 LEVIER 5 : Phase de warmup terminée ! Déblocage intégral des poids du backbone pour le Fine-Tuning.")
            for param in raw_model.backbone.parameters():
                param.requires_grad = True

        model.train()
        train_loss = 0.0
        
        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['stage2_classifier']['epochs']}"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss_elig = criterion_eligibility(outputs['eligibility'], targets)
                targets_patho = (targets > 0).long()
                loss_patho = criterion_pathology(outputs['pathology'], targets_patho)
                total_loss = loss_elig + 2.0 * loss_patho

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item()

        scheduler.step()
        train_loss /= len(train_loader) if len(train_loader) > 0 else 1

        # Phase de Validation
        model.eval()
        val_targets, val_probs = [], []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    probs = torch.softmax(outputs['pathology'], dim=1)[:, 1]
                val_targets.extend((targets > 0).cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)
        
        # Validation du Seuil Clinique
        best_threshold = 0.5
        best_metrics = {"auc_roc": 0.5, "sensitivity": 0.0, "f2_score": 0.0}
        
        if len(val_targets) > 0 and len(np.unique(val_targets)) > 1:
            for thresh in np.arange(0.1, 0.9, 0.05):
                m = calculate_clinical_metrics(val_targets, val_probs, threshold=thresh)
                if m['sensitivity'] >= 0.95 and m['f2_score'] > best_metrics['f2_score']:
                    best_metrics = m
                    best_threshold = thresh
            if best_metrics['sensitivity'] == 0.0:
                for thresh in np.arange(0.1, 0.9, 0.05):
                    m = calculate_clinical_metrics(val_targets, val_probs, threshold=thresh)
                    if m['f2_score'] > best_metrics['f2_score']:
                        best_metrics = m
                        best_threshold = thresh

        print(f"📊 Epoch {epoch} | Loss: {train_loss:.4f} | Val AUC: {best_metrics['auc_roc']:.4f} | Sensibilité: {best_metrics['sensitivity']*100:.1f}% | Score F2: {best_metrics.get('f2_score', 0):.4f} (Seuil: {best_threshold:.2f})")

        # Mise à jour des rapports d'entraînement
        reports_dir = cfg['paths']['reports_dir']
        figures_dir = cfg['paths']['figures_dir']
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

        epoch_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_auc": float(best_metrics['auc_roc']),
            "val_sensitivity": float(best_metrics['sensitivity']),
            "val_specificity": float(best_metrics.get('specificity', 0.0)),
            "val_f2_score": float(best_metrics.get('f2_score', 0.0)),
            "best_threshold": float(best_threshold)
        }

        # Sauvegarde CSV & JSON
        csv_report_path = os.path.join(reports_dir, "training_history.csv")
        df_log = pd.DataFrame([epoch_log])
        if not os.path.exists(csv_report_path):
            df_log.to_csv(csv_report_path, index=False)
        else:
            df_log.to_csv(csv_report_path, mode='a', header=False, index=False)

        import json
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

        # 🎯 LEVIER 1 : Logique d'Early Stopping & Sauvegarde des checkpoints
        if best_metrics['auc_roc'] > best_val_auc:
            best_val_auc = best_metrics['auc_roc']
            no_improve_epochs = 0
            best_path = os.path.join(cfg['paths']['checkpoints_dir'], "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'best_threshold': best_threshold,
                'val_auc': best_val_auc
            }, best_path)
            print(f"💾 LEVIER 1 : Nouveau meilleur Val AUC ({best_val_auc:.4f}) -> Modèle sauvegardé !")
        else:
            no_improve_epochs += 1
            print(f"⏳ AUC non amélioré depuis {no_improve_epochs}/{patience} epoch(s).")

        # Checkpoint de sécurité de l'état d'entraînement
        latest_dict = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_threshold': best_threshold,
            'best_val_auc': best_val_auc,
            'no_improve_epochs': no_improve_epochs
        }
        torch.save(latest_dict, latest_checkpoint_path)

        # Déclenchement de l'Early Stopping (Levier 1)
        if no_improve_epochs >= patience:
            print(f"🛑 LEVIER 1 DÉCLENCHÉ : Arrêt précoce à l'epoch {epoch} (Aucune amélioration depuis {patience} epochs).")
            print(f"🎯 Économie d'unités GPU réalisée ! Meilleur Val AUC obtenu : {best_val_auc:.4f}")
            break

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cache GPU vidé.")

if __name__ == "__main__":
    train_laafi_ai_model()
