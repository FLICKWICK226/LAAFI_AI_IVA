"""
Module de Post-Training Quantization (PTQ INT8).
Calibre le modèle Student avec 500 images représentatives et génère les poids quantisés INT8.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader, Subset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.student_model import MobileNetV4Student
from src.data.dataset import IVADataset

def quantize_student_ptq(
    model_path: str = None,
    output_path: str = "./models/exported/student_int8_ptq.pt",
    calibration_samples: int = 500,
    batch_size: int = 16
) -> str:
    """
    Exécute la Post-Training Quantization (PTQ INT8) du modèle Student.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device("cpu") # La quantification statique se fait sur CPU

    print(f"\n⚡ Lancement du Post-Training Quantization (PTQ INT8)...")
    model = MobileNetV4Student(pretrained=False, drop_rate=0.0, num_classes=1).to(device)

    if model_path and os.path.exists(model_path):
        print(f"📥 Chargement des poids FP32 : {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print("⚠️ Aucun poids spécifié. Calibration sur architecture avec poids aléatoires.")

    model.eval()

    # 1. Configuration des observateurs de quantification
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    prepared_model = quantization.prepare(model, inplace=False)

    # 2. Dataset et Dataloader de calibration (500 images)
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg['paths']['processed_data_dir']
    if os.path.exists("/kaggle/working/data/processed"):
        processed_dir = "/kaggle/working/data/processed"

    train_csv = os.path.join(processed_dir, "train.csv")
    if os.path.exists(train_csv):
        full_dataset = IVADataset(csv_file=train_csv, is_train=False)
        indices = list(range(min(calibration_samples, len(full_dataset))))
        calib_dataset = Subset(full_dataset, indices)
        calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"🧪 Calibration de la plage dynamique sur {len(calib_dataset)} images...")
        with torch.no_grad():
            for images, _, _ in calib_loader:
                prepared_model(images)
    else:
        print("⚠️ CSV de calibration introuvable. Calibration sur entrées synthétiques aléatoires.")
        for _ in range(calibration_samples // batch_size):
            dummy_input = torch.randn(batch_size, 3, 384, 384)
            prepared_model(dummy_input)

    # 3. Conversion en INT8
    print("🔄 Conversion du modèle en INT8...")
    quantized_model = quantization.convert(prepared_model, inplace=False)

    # 4. Sauvegarde
    torch.save(quantized_model.state_dict(), output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"✅ Post-Training Quantization terminée avec succès !")
    print(f"💾 Fichier INT8 généré : {output_path} ({file_size_mb:.2f} MB)")

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de quantification PTQ INT8")
    parser.add_argument("--model_path", type=str, default=None, help="Chemin du modèle FP32")
    parser.add_argument("--output_path", type=str, default="./models/exported/student_int8_ptq.pt", help="Chemin de sortie")
    parser.add_argument("--calibration_samples", type=int, default=500, help="Nombre d'images de calibration")
    args = parser.parse_args()

    quantize_student_ptq(
        model_path=args.model_path,
        output_path=args.output_path,
        calibration_samples=args.calibration_samples
    )
