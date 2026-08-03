import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.utils.seed import seed_everything
from src.data.dataset import IVADataset
from src.models.classifier_lesion import IVALesionClassifierStage2
from src.utils.metrics import FocalLoss, calculate_clinical_metrics

def train_laafi_ai_model(config_path: str = "./config/config.yaml") -> None:
    """
    Moteur d'entraînement principal pour Stage 2 (CADx).
    Exécute la précision mixte (AMP), l'optimiseur AdamW, le Cosine Scheduler
    et sauvegarde automatiquement les checkpoints sur Drive/Local.
    """
    # Chargement de la configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['project']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Exécution de l'entraînement sur : {device}")

    os.makedirs(cfg['paths']['checkpoints_dir'], exist_ok=True)
    os.makedirs(cfg['paths']['logs_dir'], exist_ok=True)

    # Chargement des Datasets et DataLoaders
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
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['stage2_classifier']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialisation du Modèle Stage 2
    model = IVALesionClassifierStage2(
        backbone_name=cfg['stage2_classifier']['backbone'],
        pretrained=True,
        num_classes_eligibility=3,
        num_classes_pathology=2
    ).to(device)

    # Loss, Optimiseur et Scheduler
    criterion_eligibility = nn.CrossEntropyLoss()
    criterion_pathology = FocalLoss(
        alpha=cfg['stage2_classifier']['focal_loss_alpha'],
        gamma=cfg['stage2_classifier']['focal_loss_gamma']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['stage2_classifier']['learning_rate']),
        weight_decay=float(cfg['stage2_classifier']['weight_decay'])
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['stage2_classifier']['epochs']
    )
    
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    best_val_auc = 0.0
    start_epoch = 1

    # -------------------------------------------------------------
    # Système Restart-Safe : Reprise Automatique depuis latest_checkpoint.pt
    # -------------------------------------------------------------
    latest_checkpoint_path = os.path.join(cfg['paths']['checkpoints_dir'], "latest_checkpoint.pt")
    if os.path.exists(latest_checkpoint_path):
        print(f"🔄 Checkpoint de reprise trouvé : {latest_checkpoint_path}")
        try:
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and scaler is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_auc = checkpoint.get('best_val_auc', 0.0)
            print(f"✅ Reprise réussie à l'epoch {start_epoch} (Dernier meilleur Val AUC : {best_val_auc:.4f})")
        except Exception as e_res:
            print(f"⚠️ Impossible de charger le checkpoint de reprise ({e_res}). Démarrage d'un nouvel entraînement.")

    if start_epoch > cfg['stage2_classifier']['epochs']:
        print(f"🎉 L'entraînement est déjà terminé ({cfg['stage2_classifier']['epochs']} epochs effectuées).")
        return

    print(f"🚀 Début/Reprise de la boucle d'entraînement de l'epoch {start_epoch} à {cfg['stage2_classifier']['epochs']}...")
    for epoch in range(start_epoch, cfg['stage2_classifier']['epochs'] + 1):
        model.train()
        train_loss = 0.0
        
        for images, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['stage2_classifier']['epochs']}"):
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss_elig = criterion_eligibility(outputs['eligibility'], targets)
                # Map target pour pathologie binaire (0: Négatif, 1: Positif)
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
                images = images.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(images)
                    probs = torch.softmax(outputs['pathology'], dim=1)[:, 1]
                val_targets.extend((targets > 0).cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)
        
        # Recherche du seuil optimal maximisant le Score F2 avec Sensibilité (Recall) >= 95%
        best_threshold = 0.5
        best_metrics = {"auc_roc": 0.5, "sensitivity": 0.0, "f2_score": 0.0}
        
        if len(val_targets) > 0 and len(np.unique(val_targets)) > 1:
            for thresh in np.arange(0.1, 0.9, 0.05):
                m = calculate_clinical_metrics(val_targets, val_probs, threshold=thresh)
                if m['sensitivity'] >= 0.95 and m['f2_score'] > best_metrics['f2_score']:
                    best_metrics = m
                    best_threshold = thresh
            # Si aucun seuil ne donne >= 95% de sensibilité, prendre le seuil maximisant le score F2
            if best_metrics['sensitivity'] == 0.0:
                for thresh in np.arange(0.1, 0.9, 0.05):
                    m = calculate_clinical_metrics(val_targets, val_probs, threshold=thresh)
                    if m['f2_score'] > best_metrics['f2_score']:
                        best_metrics = m
                        best_threshold = thresh

        print(f"📊 Epoch {epoch} | Loss: {train_loss:.4f} | Val AUC: {best_metrics['auc_roc']:.4f} | Sensibilité: {best_metrics['sensitivity']*100:.1f}% | Score F2: {best_metrics.get('f2_score', 0):.4f} (Seuil: {best_threshold:.2f})")

        # -------------------------------------------------------------
        # Enregistrement & Mise à jour dynamique du Rapport d'Entraînement après chaque Epoch
        # -------------------------------------------------------------
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

        # 1. Mise à jour du CSV d'historique
        csv_report_path = os.path.join(reports_dir, "training_history.csv")
        df_log = pd.DataFrame([epoch_log])
        if not os.path.exists(csv_report_path):
            df_log.to_csv(csv_report_path, index=False)
        else:
            df_log.to_csv(csv_report_path, mode='a', header=False, index=False)

        # 2. Mise à jour du JSON de rapport global
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

        # 3. Génération dynamique des courbes d'apprentissage (Loss & AUC)
        try:
            import matplotlib.pyplot as plt
            df_hist = pd.DataFrame(history)
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Train Loss', color='tab:red')
            ax1.plot(df_hist['epoch'], df_hist['train_loss'], color='tab:red', marker='o', label='Train Loss')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Val AUC & F2', color='tab:blue')
            ax2.plot(df_hist['epoch'], df_hist['val_auc'], color='tab:blue', marker='s', label='Val AUC')
            ax2.plot(df_hist['epoch'], df_hist['val_f2_score'], color='tab:green', marker='^', linestyle='--', label='Val F2')
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            plt.title(f"LAAFI_AI Stage 2 - Courbes d'Entraînement (Epoch {epoch})")
            fig.tight_layout()
            plt.savefig(os.path.join(figures_dir, "learning_curves.png"), dpi=150)
            plt.close()
        except Exception as e_plot:
            print(f"⚠️ Remarque lors du tracé de la courbe : {e_plot}")

        # 4. Sauvegarde systématique de latest_checkpoint.pt (Restart-Safe)
        latest_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_threshold': best_threshold,
            'best_val_auc': best_val_auc
        }
        torch.save(latest_dict, latest_checkpoint_path)
        print(f"🔄 Checkpoint de sécurité mis à jour : {latest_checkpoint_path}")

        # 5. Sauvegarde du Meilleur Checkpoint (best_model.pt) basé sur AUC & F2
        if best_metrics['auc_roc'] >= best_val_auc:
            best_val_auc = best_metrics['auc_roc']
            best_path = os.path.join(cfg['paths']['checkpoints_dir'], "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_threshold': best_threshold,
                'val_auc': best_val_auc
            }, best_path)
            print(f"💾 Nouveau meilleur modèle sauvegardé dans : {best_path}")

    # Purge VRAM en fin d'entraînement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cache GPU vidé.")

if __name__ == "__main__":
    train_laafi_ai_model()
