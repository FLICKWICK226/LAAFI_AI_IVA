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

    print("🚀 Début de la boucle d'entraînement...")
    for epoch in range(1, cfg['stage2_classifier']['epochs'] + 1):
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
        
        metrics = calculate_clinical_metrics(val_targets, val_probs, threshold=0.5) if len(val_targets) > 0 else {"auc_roc": 0.5, "sensitivity": 0.0}

        print(f"📊 Epoch {epoch} | Loss: {train_loss:.4f} | Val AUC: {metrics['auc_roc']:.4f} | Sensibilité: {metrics['sensitivity']*100:.1f}%")

        # Sauvegarde du Meilleur Checkpoint
        if metrics['auc_roc'] >= best_val_auc:
            best_val_auc = metrics['auc_roc']
            best_path = os.path.join(cfg['paths']['checkpoints_dir'], "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"💾 Nouveau meilleur modèle sauvegardé dans : {best_path}")

    # Purge VRAM en fin d'entraînement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cache GPU vidé.")

if __name__ == "__main__":
    train_laafi_ai_model()
