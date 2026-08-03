import os
import torch
from src.models.classifier_lesion import IVALesionClassifierStage2

def export_model_to_onnx(
    checkpoint_path: str = "./models/checkpoints/best_model.pt",
    output_onnx_path: str = "./models/exported/best_model.onnx",
    img_size: tuple = (384, 384)
) -> None:
    """
    Exporte le modèle Stage 2 entraîné au format ONNX pour l'inférence optimisée.
    """
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    device = torch.device("cpu")
    
    model = IVALesionClassifierStage2(pretrained=False)
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
            state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"✅ Checkpoint chargé avec succès depuis : {checkpoint_path}")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du checkpoint ({e}). Exportation de la structure du modèle.")
    else:
        print("⚠️ Aucun checkpoint trouvé, exportation du modèle non entraîné à des fins de structure.")

    model.eval()
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_crop'],
        output_names=['logits_eligibility', 'logits_pathology'],
        dynamic_axes={'input_crop': {0: 'batch_size'}}
    )

    print(f"🎉 Modèle exporté au format ONNX avec succès dans : {output_onnx_path}")

if __name__ == "__main__":
    export_model_to_onnx()
