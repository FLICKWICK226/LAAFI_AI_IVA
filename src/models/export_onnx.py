import os
import torch
import torch.nn as nn
from src.models.classifier_lesion import IVALesionClassifierStage2

class ONNXWrapper(nn.Module):
    """
    Wrapper pour convertir la sortie dictionnaire du modèle multi-tâche en tuple 
    compatible avec l'exportateur ONNX natif.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return outputs['eligibility'], outputs['pathology']

def export_model_to_onnx(
    checkpoint_path: str = "./models/checkpoints/best_model.pt",
    output_onnx_path: str = "./models/exported/best_model.onnx",
    img_size: tuple = (384, 384)
) -> None:
    """
    Exporte le modèle Stage 2 entraîné au format ONNX pour l'inférence optimisée.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_onnx_path)), exist_ok=True)
    device = torch.device("cpu")
    
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

    model = IVALesionClassifierStage2(pretrained=False)
    
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
    onnx_model = ONNXWrapper(model)
    onnx_model.eval()
    
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])

    # Utilisation de l'exportateur ONNX classique (dynamo=False pour éviter la dépendance obligatoire à onnxscript)
    export_kwargs = {
        "export_params": True,
        "opset_version": 14,
        "do_constant_folding": True,
        "input_names": ['input_crop'],
        "output_names": ['logits_eligibility', 'logits_pathology'],
        "dynamic_axes": {'input_crop': {0: 'batch_size'}}
    }
    
    # Test support du paramètre dynamo (PyTorch 2.0+)
    try:
        torch.onnx.export(onnx_model, dummy_input, output_onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(onnx_model, dummy_input, output_onnx_path, **export_kwargs)

    print(f"🎉 Modèle exporté au format ONNX avec succès dans : {output_onnx_path}")

if __name__ == "__main__":
    export_model_to_onnx()

