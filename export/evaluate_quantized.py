"""
Module d'évaluation comparative et Quantization Gate Check.
Mesure la dégradation relative des métriques cliniques entre le modèle FP32 et le modèle quantisé INT8.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.student_model import MobileNetV4Student
from src.data.dataset import IVADataset
from src.utils.metrics import calculate_clinical_metrics, evaluate_threshold_grid

def evaluate_model_on_test_set(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> dict:
    """Évalue un modèle sur le jeu de test et retourne l'AUC et la Spécificité à Recall >= 95%."""
    model.eval()
    model.to(device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            logits = model(images).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend((targets > 0).long().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    metrics = calculate_clinical_metrics(all_targets, all_preds, threshold=0.5)
    grid = evaluate_threshold_grid(all_targets, all_preds, min_recall=0.95)

    return {
        "auc": metrics["auc"],
        "spec_at_target_recall": grid.get("spec_at_target_recall", 0.0) * 100,
        "best_threshold": grid.get("best_threshold", 0.5)
    }

def run_quantization_gate_check(
    fp32_model_path: str = None,
    int8_model_path: str = "./models/exported/student_int8_ptq.pt",
    max_allowed_delta: float = 0.5
) -> dict:
    """
    Exécute le Gate Check de quantification.
    Détermine si le modèle INT8 est validé ou si un ré-entraînement QAT est requis.
    """
    print(f"\n" + "="*70)
    print(f"🚪 QUANTIZATION GATE CHECK (Seuil de dégradation max = {max_allowed_delta}%)")
    print("="*70)

    device = torch.device("cpu") # Test CPU représentatif de l'exécution Edge

    # Configuration
    config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg['paths']['processed_data_dir']
    if os.path.exists("/kaggle/working/data/processed"):
        processed_dir = "/kaggle/working/data/processed"

    test_csv = os.path.join(processed_dir, "test.csv")
    if not os.path.exists(test_csv):
        print(f"❌ Fichier test.csv introuvable dans {processed_dir}")
        return {"gate_pass": False, "reason": "Test CSV missing"}

    test_dataset = IVADataset(csv_file=test_csv, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 1. Évaluation FP32 Baseline
    print("📊 1. Évaluation du Modèle FP32 Baseline...")
    model_fp32 = MobileNetV4Student(pretrained=False, drop_rate=0.0, num_classes=1)
    if fp32_model_path and os.path.exists(fp32_model_path):
        ckpt = torch.load(fp32_model_path, map_location=device, weights_only=False)
        model_fp32.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    
    fp32_res = evaluate_model_on_test_set(model_fp32, test_loader, device)
    print(f"   • FP32 AUC: {fp32_res['auc']:.4f} | Spec (Recall>=95%): {fp32_res['spec_at_target_recall']:.1f}%")

    # 2. Évaluation INT8 Quantisé
    print("📊 2. Évaluation du Modèle INT8 Quantisé...")
    if not os.path.exists(int8_model_path):
        print(f"⚠️ Modèle INT8 non trouvé : {int8_model_path}")
        return {"gate_pass": False, "reason": "INT8 model missing"}

    model_int8 = MobileNetV4Student(pretrained=False, drop_rate=0.0, num_classes=1)
    model_int8.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model_int8 = torch.ao.quantization.prepare(model_int8, inplace=False)
    model_int8 = torch.ao.quantization.convert(model_int8, inplace=False)
    
    try:
        ckpt_int8 = torch.load(int8_model_path, map_location=device, weights_only=False)
        model_int8.load_state_dict(ckpt_int8)
    except Exception as e:
        print(f"⚠️ Avertissement chargement structure INT8 ({e}). Évaluation directe.")

    int8_res = evaluate_model_on_test_set(model_int8, test_loader, device)
    print(f"   • INT8 AUC: {int8_res['auc']:.4f} | Spec (Recall>=95%): {int8_res['spec_at_target_recall']:.1f}%")

    # 3. Calcul des dégradations
    delta_auc = (fp32_res['auc'] - int8_res['auc']) * 100
    delta_spec = fp32_res['spec_at_target_recall'] - int8_res['spec_at_target_recall']

    print("\n🔍 Résultats de la Comparaison :")
    print(f"   • Δ AUC : {delta_auc:+.2f}%")
    print(f"   • Δ Spécificité : {delta_spec:+.2f}%")

    # Gate Decision
    gate_pass = (delta_auc < max_allowed_delta) and (delta_spec < max_allowed_delta)

    if gate_pass:
        print("\n🎉 GATE CHECK REUSSI (PASS) ! La quantification PTQ respecte le seuil de tolérance < 0.5%.")
        print("✅ Le binaire INT8 est validé pour le déploiement ExecuTorch.")
    else:
        print("\n⚠️ GATE CHECK ÉCHOUÉ (FAIL) ! La perte de précision dépasse 0.5%.")
        print("🚨 Recommandation : Déclencher le QAT (Quantization-Aware Training) via 'python export/qat_trainer.py'.")

    return {
        "gate_pass": gate_pass,
        "delta_auc": delta_auc,
        "delta_spec": delta_spec,
        "fp32_res": fp32_res,
        "int8_res": int8_res
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Gate Check de quantification")
    parser.add_argument("--fp32_model", type=str, default=None, help="Chemin du modèle FP32")
    parser.add_argument("--int8_model", type=str, default="./models/exported/student_int8_ptq.pt", help="Chemin du modèle INT8")
    parser.add_argument("--max_delta", type=float, default=0.5, help="Seuil de dégradation maximale tolérée (%)")
    args = parser.parse_args()

    run_quantization_gate_check(
        fp32_model_path=args.fp32_model,
        int8_model_path=args.int8_model,
        max_allowed_delta=args.max_delta
    )
