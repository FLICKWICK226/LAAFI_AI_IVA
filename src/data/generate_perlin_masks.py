import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_coherent_noise_mask(shape=(384, 384), octaves=3, persistence=0.5) -> np.ndarray:
    """
    Génère un masque de bruit procédural multi-échelles cohérent (fractal noise)
    vectorisé sous NumPy + OpenCV sans dépendance C externe.
    """
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    freq = 1.0
    amp = 1.0
    total_amp = 0.0

    for _ in range(octaves):
        grid_h = max(2, int(h / (48 / freq)))
        grid_w = max(2, int(w / (48 / freq)))
        noise_grid = np.random.randn(grid_h, grid_w).astype(np.float32)
        upscaled = cv2.resize(noise_grid, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * upscaled
        total_amp += amp
        amp *= persistence
        freq *= 2.0

    out /= (total_amp + 1e-8)
    p_min, p_max = out.min(), out.max()
    out = (out - p_min) / (p_max - p_min + 1e-8)
    return out.astype(np.float32)

def generate_perlin_masks(
    output_dir: str = "./data/synthetic_masks",
    num_masks: int = 1000,
    shape: tuple = (384, 384),
    scale: float = 100.0,
    octaves: int = 3,
    persistence: float = 0.5,
    lacunarity: float = 2.0
) -> None:
    """
    Génère et sauvegarde hors-ligne 'num_masks' masques de bruit procédural au format .npy
    (100x plus rapide et zéro dépendance externe).
    """
    # Auto-détection de l'environnement Kaggle / Colab
    if os.path.exists("/kaggle/working"):
        output_dir = "/kaggle/working/data/synthetic_masks"

    os.makedirs(output_dir, exist_ok=True)
    h, w = shape
    
    existing_masks = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
    if len(existing_masks) >= num_masks:
        print(f"✅ {len(existing_masks)} masques procéduraux déjà disponibles dans : {output_dir}")
        return

    print(f"🎨 Génération hors-ligne vectorisée de {num_masks} masques ({h}x{w})...")
    
    for k in tqdm(range(len(existing_masks), num_masks), desc="Génération Masques"):
        mask = generate_coherent_noise_mask(shape=shape, octaves=octaves, persistence=persistence)
        mask_path = os.path.join(output_dir, f"perlin_mask_{k:04d}.npy")
        np.save(mask_path, mask)

    print(f"🎉 {num_masks} masques de bruit procédural générés dans : {output_dir}")

if __name__ == "__main__":
    generate_perlin_masks()

