import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def generate_gradcam_heatmap(model: torch.nn.Module, image_tensor: torch.tensor, target_layer: torch.nn.Module) -> np.ndarray:
    """
    Génère une heatmap Grad-CAM pour vérifier que le réseau focalise son attention
    sur les zones acéto-blanches de la JSC et non sur le métal ou le sang (Rule 6).
    """
    model.eval()
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    # Passage avant
    outputs = model(image_tensor.unsqueeze(0))
    target_score = outputs['pathology'][0, 1]

    # Rétropropagation
    model.zero_grad()
    target_score.backward()

    hook_handle.remove()

    if len(gradients) == 0 or len(activations) == 0:
        return np.zeros((384, 384), dtype=np.float32)

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activation = activations[0][0]

    for i in range(activation.shape[0]):
        activation[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    
    return cv2.resize(heatmap, (384, 384))

def save_gradcam_audit_figure(image_np: np.ndarray, heatmap: np.ndarray, save_path: str):
    """
    Superpose la heatmap Grad-CAM sur l'image d'origine et la sauvegarde pour audit visuel.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
