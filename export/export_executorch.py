"""
Module d'exportation du graphe PyTorch vers ExecuTorch (.pt2) et ONNX (.onnx).
Garantit une entrée fixe [1, 3, 384, 384] et un graphe statique sans opérations dynamiques.
"""

import os
import sys
import argparse
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.student_model import MobileNetV4Student

def export_student_to_executorch(
    model_path: str = None,
    output_dir: str = "./models/exported",
    img_size: int = 384
) -> dict:
    """
    Exporte le modèle Student MobileNetV4 vers ExecuTorch (.pt2) et ONNX (.onnx).
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    print(f"📱 Initialisation du modèle Student pour l'export ExecuTorch/ONNX...")
    model = MobileNetV4Student(pretrained=False, drop_rate=0.0, num_classes=1)

    if model_path and os.path.exists(model_path):
        print(f"📥 Chargement des poids depuis {model_path}")
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        print("⚠️ Aucun poids spécifié. Export d'un modèle non-entraîné à des fins de structure.")

    model.eval()

    example_input = torch.randn(1, 3, img_size, img_size, device=device)

    # 1. Export ExecuTorch via torch.export
    pt2_path = os.path.join(output_dir, "laafi_student_384.pt2")
    print(f"🚀 Export ExecuTorch via torch.export.export() avec entrée ({1}, {3}, {img_size}, {img_size})...")
    try:
        exported_program = torch.export.export(model, (example_input,))
        torch.export.save(exported_program, pt2_path)
        pt2_size = os.path.getsize(pt2_path) / (1024 * 1024)
        print(f"✅ Graphe PyTorch exporté avec succès : {pt2_path} ({pt2_size:.2f} MB)")
    except Exception as e:
        print(f"⚠️ Erreur lors de l'export torch.export ({e}). Fallback vers ONNX.")
        pt2_path = None

    # 2. Export ONNX (Format alternatif universel pour Android / ONNX Runtime ARM64)
    onnx_path = os.path.join(output_dir, "laafi_student_384.onnx")
    print(f"📦 Export ONNX (opset=17)...")
    try:
        torch.onnx.export(
            model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=['pathology_logits']
        )
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"✅ Modèle ONNX exporté avec succès : {onnx_path} ({onnx_size:.2f} MB)")
    except Exception as e:
        print(f"❌ Erreur lors de l'export ONNX : {e}")
        onnx_path = None

    return {
        "pt2_path": pt2_path,
        "onnx_path": onnx_path
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'export ExecuTorch & ONNX")
    parser.add_argument("--model_path", type=str, default=None, help="Chemin du checkpoint PyTorch (.pt)")
    parser.add_argument("--output_dir", type=str, default="./models/exported", help="Dossier de sortie")
    args = parser.parse_args()

    export_student_to_executorch(model_path=args.model_path, output_dir=args.output_dir)
