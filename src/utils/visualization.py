import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def generate_gradcam_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module = None,
    output_path: str = None
) -> np.ndarray:
    """
    Génère une heatmap Grad-CAM pour vérifier que le réseau focalise son attention
    sur les zones acéto-blanches de la JSC et non sur le métal ou le sang (Rule 6).
    """
    if model is None:
        # Fallback dummy heatmap if model is not loaded yet
        heatmap = np.zeros((384, 384), dtype=np.float32)
        if output_path:
            img_np = np.zeros((384, 384, 3), dtype=np.uint8)
            save_gradcam_audit_figure(img_np, heatmap, output_path)
        return heatmap

    model.eval()

    # Détection automatique de la couche cible si non spécifiée
    if target_layer is None:
        if hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
            target_layer = model.backbone.stages[-1]
        elif hasattr(model, "backbone") and hasattr(model.backbone, "features"):
            target_layer = model.backbone.features[-1]
        else:
            # Recherche de la dernière couche Conv2d dans le modèle
            conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
            if conv_layers:
                target_layer = conv_layers[-1]

    if target_layer is None:
        heatmap = np.zeros((384, 384), dtype=np.float32)
        if output_path:
            img_np = np.zeros((384, 384, 3), dtype=np.uint8)
            save_gradcam_audit_figure(img_np, heatmap, output_path)
        return heatmap

    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    # Passage avant (batch dim = 1)
    if image_tensor.dim() == 3:
        input_tensor = image_tensor.unsqueeze(0)
    else:
        input_tensor = image_tensor

    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    outputs = model(input_tensor)
    if isinstance(outputs, dict) and 'pathology' in outputs:
        target_score = outputs['pathology'][0, 1]
    elif isinstance(outputs, torch.Tensor):
        target_score = outputs[0, 1] if outputs.dim() > 1 else outputs[0]
    else:
        target_score = outputs[0]

    # Rétropropagation
    model.zero_grad()
    target_score.backward()

    hook_handle.remove()

    if len(gradients) == 0 or len(activations) == 0:
        heatmap = np.zeros((384, 384), dtype=np.float32)
    else:
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
        activation = activations[0][0]

        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activation, dim=0).detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)
        heatmap = cv2.resize(heatmap, (384, 384))

    # Sauvegarde automatique si output_path est fourni
    if output_path:
        # Reconstitution d'une image uint8 RGB à partir d'image_tensor
        img_np = image_tensor.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # Dé-normalisation approximative ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

        save_gradcam_audit_figure(img_np, heatmap, output_path)

    return heatmap

def save_gradcam_audit_figure(image_np: np.ndarray, heatmap: np.ndarray, save_path: str):
    """
    Superpose la heatmap Grad-CAM sur l'image d'origine et la sauvegarde pour audit visuel.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed = cv2.addWeighted(image_np, 0.6, heatmap_colored, 0.4, 0)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Image d'Origine (Crop JSC)")
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title("Carte d'Attention Grad-CAM")
    axes[1].axis('off')
    
    axes[2].imshow(superimposed)
    axes[2].set_title("Superposition Clinique")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

