import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ajout du chemin projet au sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.seed import seed_everything
from src.data.dataset import IVADataset
from src.models.classifier_lesion import IVALesionClassifierStage2
from src.utils.metrics import FocalLoss, calculate_clinical_metrics, evaluate_threshold_grid
from src.losses.asymmetric_loss import AsymmetricFocalLoss

def run_ablation_experiment(
    loss_type: str = "asymmetric",  # "focal" ou "asymmetric"
    epochs: int = 15,
    seed: int = 42
) -> dict:
    """
    Script d'ablation comparant la Focal Loss classique (alpha=0.75) avec l'Asymmetric Loss (ASL).
    Évalue la spécificité obtenue au seuil calibré (Recall >= 95%) sur val.csv et test.csv.
    """
    print(f"\n" + "="*70)
    print(f"🔬 Lancement de l'Ablation avec Loss: '{loss_type.upper()}' (Seed={seed}, Epochs={epochs})")
    print("="*70)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Périphérique de calcul : {device}")

    # Configuration
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Instantier la loss sélectionnée
    if loss_type == "asymmetric":
        criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0, clip=0.05)
    else:
        criterion = FocalLoss(alpha=0.75, gamma=2.0)

    # Instantier le modèle ConvNeXt-Base
    model = IVALesionClassifierStage2(
        backbone_name=cfg['stage2_classifier']['backbone'],
        pretrained=True,
        num_classes_eligibility=3,
        num_classes_pathology=2,
        drop_rate=float(cfg['stage2_classifier'].get('drop_rate', 0.4))
    ).to(device)

    # Optimizer avec weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg['stage2_classifier']['learning_rate']),
        weight_decay=float(cfg['stage2_classifier']['weight_decay'])
    )

    # Chargement des Datasets
    processed_dir = cfg['paths']['processed_data_dir']
    if os.path.exists("/kaggle/working/data/processed"):
        processed_dir = "/kaggle/working/data/processed"

    train_csv = os.path.join(processed_dir, "train.csv")
    val_csv = os.path.join(processed_dir, "val.csv")
    test_csv = os.path.join(processed_dir, "test.csv")

    if not os.path.exists(train_csv):
        print(f"⚠️ Dataset non trouvé dans {processed_dir}. Exécutez d'abord la préparation des données.")
        return {}

    train_ds = IVADataset(csv_file=train_csv, is_train=True)
    val_ds = IVADataset(csv_file=val_csv, is_train=False)
    test_ds = IVADataset(csv_file=test_csv, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    best_val_auc = 0.0
    best_threshold = 0.38

    # Boucle d'entraînement
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets, _ in tqdm(train_loader, desc=f"Époque {epoch}/{epochs} [{loss_type.upper()}]"):
            images = images.to(device)
            targets_patho = (targets > 0).long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs['pathology']
            loss = criterion(logits, targets_patho)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Évaluation Validation
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs['pathology'], dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend((targets > 0).cpu().numpy())

        val_probs = np.array(val_probs)
        val_targets = np.array(val_targets)

        # Calibration du seuil T sur Validation (Recall >= 95%)
        grid_eval = evaluate_threshold_grid(val_targets, val_probs)
        opt_res = grid_eval['optimal']

        if opt_res['auc_roc'] > best_val_auc:
            best_val_auc = opt_res['auc_roc']
            best_threshold = opt_res['threshold']

        print(f"Époque {epoch:02d} | Loss: {running_loss/len(train_loader):.4f} | Val AUC: {opt_res['auc_roc']:.4f} | Val Spec (Recall>=95%): {opt_res['specificity']*100:.1f}% | Seuil: {opt_res['threshold']:.2f}")

    # Évaluation finale sur Test Set à aveugle avec le seuil gelé
    model.eval()
    test_probs, test_targets = [], []
    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs['pathology'], dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend((targets > 0).cpu().numpy())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)

    final_metrics = calculate_clinical_metrics(test_targets, test_probs, threshold=best_threshold)
    print("\n" + "="*70)
    print(f"📊 RÉSULTATS TEST FINAL ({loss_type.upper()}) | Seuil Gelé T = {best_threshold:.2f}")
    print(f"  • Sensibilité (Recall) : {final_metrics['sensitivity']*100:.1f}%")
    print(f"  • Spécificité          : {final_metrics['specificity']*100:.1f}%")
    print(f"  • Score F2             : {final_metrics['f2_score']:.4f}")
    print(f"  • AUC-ROC              : {final_metrics['auc_roc']:.4f}")
    print("="*70)

    return final_metrics

if __name__ == "__main__":
    run_ablation_experiment(loss_type="asymmetric", epochs=15)
