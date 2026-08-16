import os
import sys
import importlib
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ajout du chemin projet au sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Force-reload des modules critiques pour éviter le cache noyau Jupyter/Kaggle
import src.losses.asymmetric_loss
import src.utils.metrics
import src.data.dataset
import src.models.classifier_lesion
importlib.reload(src.losses.asymmetric_loss)
importlib.reload(src.utils.metrics)
importlib.reload(src.data.dataset)
importlib.reload(src.models.classifier_lesion)

from src.utils.seed import seed_everything
from src.data.dataset import IVADataset
from src.models.classifier_lesion import IVALesionClassifierStage2
from src.utils.metrics import FocalLoss, calculate_clinical_metrics, evaluate_threshold_grid
from src.losses.asymmetric_loss import AsymmetricFocalLoss


def _get_ablation_ckpt_path(loss_type: str) -> str:
    """Retourne le chemin de checkpoint de reprise selon l'environnement."""
    base = "/kaggle/working/models/checkpoints" if os.path.exists("/kaggle/working") else "./models/checkpoints"
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"ablation_{loss_type}_resume.pt")


def run_ablation_experiment(
    loss_type: str = "asymmetric",  # "focal" ou "asymmetric"
    epochs: int = 15,
    seed: int = 42
) -> dict:
    """
    Script d'ablation comparant la Focal Loss classique (alpha=0.75) avec l'Asymmetric Loss (ASL).
    Évalue la spécificité obtenue au seuil calibré (Recall >= 95%) sur val.csv et test.csv.
    Supporte la reprise automatique après déconnexion (checkpoint par époque).
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

    # Instantier les loss pour warmup et régime permanent
    criterion_warmup = nn.CrossEntropyLoss()
    criterion_asl = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0, clip=0.05)
    criterion_focal = FocalLoss(alpha=0.75, gamma=2.0)

    # Instantier le modèle (convnext_small par défaut ou configuré)
    model = IVALesionClassifierStage2(
        backbone_name=cfg['stage2_classifier'].get('backbone', 'convnext_small'),
        pretrained=True,
        num_classes_eligibility=3,
        num_classes_pathology=2,
        drop_rate=float(cfg['stage2_classifier'].get('drop_rate', 0.3))
    ).to(device)

    # Optimizer avec Differential Learning Rate (Backbone à 1e-4, Têtes à 1e-3)
    backbone_lr = float(cfg['stage2_classifier'].get('learning_rate', 1e-4))
    head_lr = float(cfg['stage2_classifier'].get('head_learning_rate', 1e-3))
    weight_decay = float(cfg['stage2_classifier'].get('weight_decay', 1e-4))

    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': backbone_lr},
        {'params': model.head_pathology.parameters(), 'lr': head_lr},
        {'params': model.head_eligibility.parameters(), 'lr': head_lr}
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── Reprise automatique depuis checkpoint ────────────────────────────────
    ckpt_path = _get_ablation_ckpt_path(loss_type)
    start_epoch = 1
    best_val_auc = 0.0
    best_threshold = 0.38

    if os.path.exists(ckpt_path):
        print(f"♻️  Checkpoint de reprise détecté : {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_val_auc = ckpt.get('best_val_auc', 0.0)
            best_threshold = ckpt.get('best_threshold', 0.38)
            print(f"✅ Reprise depuis l'époque {start_epoch} | Meilleur AUC : {best_val_auc:.4f} | Seuil : {best_threshold:.2f}")
        except Exception as e:
            print(f"⚠️  Checkpoint corrompu ({e}). Démarrage depuis l'époque 1.")
            start_epoch = 1
    else:
        print("🆕 Aucun checkpoint trouvé — démarrage depuis l'époque 1.")

    if start_epoch > epochs:
        print(f"🎉 Entraînement déjà terminé ({epochs} époques). Passage direct à l'évaluation Test.")
    # ────────────────────────────────────────────────────────────────────────

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

    batch_size = int(cfg['stage2_classifier'].get('batch_size', 16))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # Boucle d'entraînement avec reprise
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0

        # Stratégie de Loss Warmup :
        # Époques 1-3 : CrossEntropy (gradients pleins pour ancrer les poids du backbone)
        # Époques 4+ : AsymmetricFocalLoss (spécificité maximale sans faux positifs)
        if loss_type == "asymmetric":
            if epoch <= 3:
                active_criterion = criterion_warmup
                loss_name = "WARMUP_CE"
            else:
                active_criterion = criterion_asl
                loss_name = "ASYMMETRIC"
        else:
            active_criterion = criterion_focal
            loss_name = "FOCAL"

        for images, targets, _ in tqdm(train_loader, desc=f"Époque {epoch}/{epochs} [{loss_name}]"):
            images = images.to(device, non_blocking=True)
            targets_patho = (targets > 0).long().to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                logits = outputs['pathology']
                loss = active_criterion(logits, targets_patho)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        # Évaluation Validation
        model.eval()
        val_probs, val_targets_list = [], []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs['pathology'], dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets_list.extend((targets > 0).cpu().numpy())

        val_probs = np.array(val_probs)
        val_targets_arr = np.array(val_targets_list)

        # Calibration du seuil T sur Validation (Recall >= 95%)
        grid_eval = evaluate_threshold_grid(val_targets_arr, val_probs)
        opt_res = grid_eval['optimal']

        if opt_res['auc_roc'] > best_val_auc:
            best_val_auc = opt_res['auc_roc']
            best_threshold = opt_res['threshold']

        epoch_log = (
            f"Époque {epoch:02d} | Loss: {running_loss/len(train_loader):.4f} "
            f"| Val AUC: {opt_res['auc_roc']:.4f} "
            f"| Val Spec (Recall>=95%): {opt_res['specificity']*100:.1f}% "
            f"| Seuil: {opt_res['threshold']:.2f}"
        )
        print(epoch_log)

        # ── Sauvegarde checkpoint de reprise (écrasement à chaque époque) ────
        torch.save({
            'epoch': epoch,
            'loss_type': loss_type,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
            'best_threshold': best_threshold,
        }, ckpt_path)
        print(f"💾 Checkpoint sauvegardé → {ckpt_path} (reprise possible depuis l'époque {epoch + 1})")
        # ────────────────────────────────────────────────────────────────────

    # Évaluation finale sur Test Set à aveugle avec le seuil gelé
    model.eval()
    test_probs, test_targets_list = [], []
    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs['pathology'], dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_targets_list.extend((targets > 0).cpu().numpy())

    test_probs = np.array(test_probs)
    test_targets_arr = np.array(test_targets_list)

    final_metrics = calculate_clinical_metrics(test_targets_arr, test_probs, threshold=best_threshold)
    print("\n" + "="*70)
    print(f"📊 RÉSULTATS TEST FINAL ({loss_type.upper()}) | Seuil Gelé T = {best_threshold:.2f}")
    print(f"  • Sensibilité (Recall) : {final_metrics['sensitivity']*100:.1f}%")
    print(f"  • Spécificité          : {final_metrics['specificity']*100:.1f}%")
    print(f"  • Score F2             : {final_metrics['f2_score']:.4f}")
    print(f"  • AUC-ROC              : {final_metrics['auc_roc']:.4f}")
    print("="*70)

    # Nettoyage du checkpoint de reprise après succès
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"🗑️  Checkpoint de reprise supprimé (run terminé avec succès).")

    return final_metrics


if __name__ == "__main__":
    run_ablation_experiment(loss_type="asymmetric", epochs=15)
