"""
Script d'exécution de la Distillation Hybride (Teacher ConvNeXt-Base ➔ Student MobileNetV4-Small).
Conforme au plan d'ingénierie Edge AI de la Semaine 2.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.seed import seed_everything
from src.data.dataset import IVADataset
from src.models.classifier_lesion import IVALesionClassifierStage2
from src.models.student_model import MobileNetV4Student
from src.distillation.kd_loss import BinaryHybridKDLoss
from src.utils.metrics import calculate_clinical_metrics, evaluate_threshold_grid

def get_teacher_checkpoint_path() -> str:
    """Trouve le meilleur checkpoint du Teacher ConvNeXt-Base."""
    candidates = [
        "/kaggle/working/models/checkpoints/ablation_asymmetric_resume.pt",
        "./models/checkpoints/ablation_asymmetric_resume.pt",
        "./models/checkpoints/teacher_convnext_best.pt"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def run_distillation_experiment(
    epochs: int = 15,
    batch_size: int = 16,
    seed: int = 42,
    teacher_ckpt: str = None
) -> dict:
    """
    Lance le pipeline de distillation hybride Teacher ➔ Student.
    """
    print(f"\n" + "="*75)
    print(f"🎓 Lancement de la Distillation Hybride (Teacher ConvNeXt ➔ Student MobileNetV4)")
    print(f"   Seed={seed} | Epochs={epochs} | BatchSize={batch_size}")
    print("="*75)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Périphérique de calcul : {device}")

    # Configuration
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Chargement du Teacher (Gelé)
    if teacher_ckpt is None:
        teacher_ckpt = get_teacher_checkpoint_path()

    print(f"🔍 Chargement du Teacher ConvNeXt-Base...")
    teacher = IVALesionClassifierStage2(
        backbone_name=cfg['stage2_classifier']['backbone'],
        pretrained=True,
        num_classes_eligibility=3,
        num_classes_pathology=2
    ).to(device)

    if teacher_ckpt and os.path.exists(teacher_ckpt):
        print(f"✅ Checkpoint Teacher trouvé : {teacher_ckpt}")
        ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            teacher.load_state_dict(ckpt['model_state_dict'])
        else:
            teacher.load_state_dict(ckpt)
    else:
        print("⚠️ Aucun checkpoint Teacher spécifique trouvé. Utilisation des poids pré-entraînés par défaut.")

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # 2. Instantier le Student (MobileNetV4-Small)
    print("📱 Instantier le Student MobileNetV4-Small...")
    student = MobileNetV4Student(
        backbone_name="mobilenetv4_conv_small",
        pretrained=True,
        drop_rate=0.2,
        num_classes=1
    ).to(device)

    # 3. Criterion KD Hybride
    kd_criterion = BinaryHybridKDLoss(
        alpha_kd=0.6,
        beta_attn=50.0,
        temperature=4.0,
        use_asymmetric_hard_loss=True
    )

    # 4. Optimizer avec LLRD (Layer-wise Learning Rate Decay)
    backbone_params = [p for n, p in student.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in student.named_parameters() if "classifier" in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-4}
    ], weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Checkpoint de reprise pour le Student
    base_ckpt_dir = "/kaggle/working/models/checkpoints" if os.path.exists("/kaggle/working") else "./models/checkpoints"
    os.makedirs(base_ckpt_dir, exist_ok=True)
    student_resume_ckpt = os.path.join(base_ckpt_dir, "distillation_student_resume.pt")
    student_best_ckpt = os.path.join(base_ckpt_dir, "student_mobilenetv4_best.pt")

    start_epoch = 1
    best_val_auc = 0.0

    if os.path.exists(student_resume_ckpt):
        print(f"♻️ Checkpoint de reprise Student trouvé : {student_resume_ckpt}")
        try:
            ckpt_s = torch.load(student_resume_ckpt, map_location=device, weights_only=False)
            student.load_state_dict(ckpt_s['model_state_dict'])
            optimizer.load_state_dict(ckpt_s['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt_s['scheduler_state_dict'])
            start_epoch = ckpt_s['epoch'] + 1
            best_val_auc = ckpt_s.get('best_val_auc', 0.0)
            print(f"✅ Reprise Student depuis époque {start_epoch} | Meilleur AUC : {best_val_auc:.4f}")
        except Exception as e:
            print(f"⚠️ Erreur chargement checkpoint Student ({e}). Démarrage époque 1.")

    # 5. Datasets & DataLoaders
    processed_dir = cfg['paths']['processed_data_dir']
    if os.path.exists("/kaggle/working/data/processed"):
        processed_dir = "/kaggle/working/data/processed"

    train_dataset = IVADataset(csv_file=os.path.join(processed_dir, "train.csv"), is_train=True)
    val_dataset = IVADataset(csv_file=os.path.join(processed_dir, "val.csv"), is_train=False)
    test_dataset = IVADataset(csv_file=os.path.join(processed_dir, "test.csv"), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Hook d'extraction de la carte d'attention du Teacher
    teacher_feat_map = [None]
    def teacher_hook(module, input, output):
        teacher_feat_map[0] = output

    # Hook sur la dernière couche de conv du Teacher ConvNeXt
    hook_handle = None
    if hasattr(teacher.backbone, 'feature_info'):
        # Attacher le hook à la dernière couche avant pooling
        for name, module in teacher.backbone.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
                last_module = module
        if last_module:
            hook_handle = last_module.register_forward_hook(teacher_hook)

    print("\n🚀 Début du cycle de Distillation Student...")
    for epoch in range(start_epoch, epochs + 1):
        student.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Époque {epoch:02d}/{epochs:02d} [KD Student]", leave=True)
        for images, targets, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # Inférence Teacher (sans calcul de gradients)
            with torch.no_grad():
                teacher_out = teacher(images)
                t_logits = teacher_out["pathology"][:, 1] if teacher_out["pathology"].ndim > 1 else teacher_out["pathology"]

            # Inférence Student
            s_logits = student(images).squeeze(-1)
            s_feat = student.get_last_feature_map()
            t_feat = teacher_feat_map[0]

            # Binary Targets pour IVA pathology (0=Type 1/2 Neg, 1=Type 3 ou Positif)
            b_targets = (targets > 0).long()

            loss = kd_criterion(
                student_logits=s_logits,
                teacher_logits=t_logits,
                student_feats=s_feat,
                teacher_feats=t_feat,
                targets=b_targets
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        scheduler.step()
        epoch_loss = running_loss / max(1, len(train_loader))

        # Évaluation Validation
        student.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                logits = student(images).squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend((targets > 0).long().numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        val_metrics = calculate_clinical_metrics(val_targets, val_preds, threshold=0.5)
        grid_metrics = evaluate_threshold_grid(val_targets, val_preds, min_recall=0.95)

        val_auc = val_metrics['auc']
        val_spec = grid_metrics.get('spec_at_target_recall', 0.0) * 100
        best_t = grid_metrics.get('best_threshold', 0.5)

        print(f"Époque {epoch:02d} | Loss KD: {epoch_loss:.4f} | Val AUC: {val_auc:.4f} | Val Spec (Recall>=95%): {val_spec:.1f}% | Seuil: {best_t:.2f}")

        # Sauvegarde Checkpoints
        ckpt_data = {
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': max(best_val_auc, val_auc),
            'best_threshold': best_t
        }
        torch.save(ckpt_data, student_resume_ckpt)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(ckpt_data, student_best_ckpt)
            print(f"🌟 Nouveau meilleur Student FP32 sauvegardé ! (Val AUC: {val_auc:.4f})")

    if hook_handle is not None:
        hook_handle.remove()

    print("\n✅ Distillation terminée avec succès.")
    return {"best_val_auc": best_val_auc}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de distillation MobileNetV4 Student")
    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=16, help="Taille des mini-lots")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--teacher_ckpt", type=str, default=None, help="Chemin du checkpoint Teacher")
    args = parser.parse_args()

    run_distillation_experiment(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        teacher_ckpt=args.teacher_ckpt
    )
