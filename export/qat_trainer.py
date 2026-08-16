"""
Module de Quantization-Aware Training (QAT).
Ré-entraîne le modèle Student pendant 3 à 5 époques avec insertion de fake-quantization nodes
en cas d'échec du PTQ (dégradation de précision >= 0.5%).
"""

import os
import sys
import argparse
import yaml
import torch
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.student_model import MobileNetV4Student
from src.data.dataset import IVADataset
from src.losses.asymmetric_loss import AsymmetricFocalLoss

def train_student_qat(
    fp32_model_path: str = None,
    output_path: str = "./models/exported/student_int8_qat.pt",
    epochs: int = 5,
    lr: float = 1e-5,
    batch_size: int = 16
) -> str:
    """
    Exécute le ré-entraînement Quantization-Aware Training (QAT).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device("cpu") # QAT PyTorch standard s'exécute sur CPU / CUDA avec qconfig fbgemm

    print(f"\n" + "="*70)
    print(f"🔄 Déclenchement du Quantization-Aware Training (QAT) — {epochs} Époques")
    print("="*70)

    # 1. Charger le modèle FP32
    model = MobileNetV4Student(pretrained=False, drop_rate=0.1, num_classes=1).to(device)
    if fp32_model_path and os.path.exists(fp32_model_path):
        ckpt = torch.load(fp32_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)

    # 2. Préparer pour QAT (insertion des fake-quantization nodes)
    model.train()
    model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    qat_model = quantization.prepare_qat(model, inplace=False)

    # 3. DataLoaders
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg['paths']['processed_data_dir']
    if os.path.exists("/kaggle/working/data/processed"):
        processed_dir = "/kaggle/working/data/processed"

    train_dataset = IVADataset(csv_file=os.path.join(processed_dir, "train.csv"), is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(qat_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0)

    # 4. Boucle d'entraînement QAT
    print("🚀 Début du fine-tuning QAT avec fake-quantization...")
    for epoch in range(1, epochs + 1):
        qat_model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Époque {epoch:02d}/{epochs:02d} [QAT Fine-Tuning]")
        for images, targets, _ in pbar:
            images = images.to(device)
            targets = (targets > 0).float().to(device)

            logits = qat_model(images).squeeze(-1)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    # 5. Conversion en INT8 quantisé final
    print("\n🔄 Conversion finale du modèle QAT en poids INT8...")
    qat_model.eval()
    quantized_model = quantization.convert(qat_model, inplace=False)

    torch.save(quantized_model.state_dict(), output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"✅ Quantization-Aware Training terminé avec succès !")
    print(f"💾 Binaire INT8 QAT généré : {output_path} ({file_size_mb:.2f} MB)")

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de QAT (Quantization-Aware Training)")
    parser.add_argument("--fp32_model", type=str, default=None, help="Chemin du modèle FP32")
    parser.add_argument("--output_path", type=str, default="./models/exported/student_int8_qat.pt", help="Chemin du fichier de sortie")
    parser.add_argument("--epochs", type=int, default=5, help="Nombre d'époques QAT")
    parser.add_argument("--lr", type=float, default=1e-5, help="Taux d'apprentissage")
    args = parser.parse_args()

    train_student_qat(
        fp32_model_path=args.fp32_model,
        output_path=args.output_path,
        epochs=args.epochs,
        lr=args.lr
    )
