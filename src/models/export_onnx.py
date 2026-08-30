import os
import torch
import torch.nn as nn
from src.models.classifier_lesion import IVALesionClassifierStage2

def export_model_to_onnx(
    checkpoint_path: str = "./models/checkpoints/best_model.pt",
    output_onnx_path: str = "./models/exported/best_model.onnx",
    img_size: tuple = (224, 224),
    backbone_name: str = None,
    ckpt_path: str = None
) -> None:
    """
    Exporte le modèle Stage 2 Unifié (Single-Head 3-Classes) au format ONNX pour inférence mobile.
    """
    if checkpoint_path == "./models/checkpoints/best_model.pt" and ckpt_path is not None:
        checkpoint_path = ckpt_path
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    device = torch.device("cpu")
    
    # Résolution dynamique du backbone depuis la config si non spécifié
    if backbone_name is None:
        backbone_name = "convnext_small"
        config_candidates = [
            "./config/config.yaml",
            "/kaggle/working/LAAFI_AI_IVA/config/config.yaml",
            "../config/config.yaml"
        ]
        for cfg_path in config_candidates:
            if os.path.exists(cfg_path):
                try:
                    import yaml
                    with open(cfg_path, 'r', encoding='utf-8') as f_cfg:
                        cfg = yaml.safe_load(f_cfg)
                        backbone_name = cfg.get('stage2_classifier', {}).get('backbone', 'convnext_small')
                        break
                except Exception:
                    pass

    # Résolution multi-chemins du checkpoint
    candidate_paths = [
        checkpoint_path,
        "/kaggle/working/models/checkpoints/best_model.pt",
        "./models/checkpoints/best_model.pt",
        "../models/checkpoints/best_model.pt"
    ]
    
    resolved_checkpoint = None
    for path in candidate_paths:
        if path and os.path.exists(path):
            resolved_checkpoint = path
            break

    model = IVALesionClassifierStage2(backbone_name=backbone_name, pretrained=False, num_classes=3)
    
    if resolved_checkpoint:
        try:
            try:
                ckpt = torch.load(resolved_checkpoint, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(resolved_checkpoint, map_location=device)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            
            # Nettoyage des clés compilées torch.compile() (_orig_mod.)
            clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
            print(f"✅ Checkpoint chargé avec succès depuis : {resolved_checkpoint}")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du checkpoint ({e}). Exportation de la structure du modèle.")
    else:
        print("⚠️ Aucun checkpoint trouvé, exportation du modèle non entraîné à des fins de structure.")

    model.eval()
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])

    export_kwargs = {
        "export_params": True,
        "opset_version": 14,
        "do_constant_folding": True,
        "input_names": ['input_crop'],
        "output_names": ['logits'],
        "dynamic_axes": {
            'input_crop': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    }
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            dynamo=False,
            **export_kwargs
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx_path,
            **export_kwargs
        )

    print(f"🎉 Modèle Stage 2 Unifié exporté avec succès en ONNX -> {output_onnx_path}")

if __name__ == "__main__":
    export_model_to_onnx()
