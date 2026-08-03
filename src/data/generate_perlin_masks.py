import os
import numpy as np
import noise
from tqdm import tqdm

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
    Génère et sauvegarde hors-ligne 'num_masks' masques de bruit de Perlin au format .npy.
    Cela élimine 100% du goulot d'étranglement CPU pendant l'entraînement PyTorch.
    """
    os.makedirs(output_dir, exist_ok=True)
    h, w = shape
    
    # Vérification si les masques existent déjà
    existing_masks = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
    if len(existing_masks) >= num_masks:
        print(f"✅ {len(existing_masks)} masques de Perlin déjà disponibles dans : {output_dir}")
        return

    print(f"🎨 Génération hors-ligne de {num_masks} masques de Perlin ({h}x{w})...")
    
    for k in tqdm(range(len(existing_masks), num_masks), desc="Génération Masques"):
        perlin_map = np.zeros((h, w), dtype=np.float32)
        seed = np.random.randint(0, 10000)
        
        # Grille d'indices pour accélérer l'évaluation
        for i in range(h):
            for j in range(w):
                perlin_map[i, j] = noise.pnoise2(
                    i / scale,
                    j / scale,
                    octaves=octaves,
                    persistence=persistence,
                    lacunarity=lacunarity,
                    base=seed
                )
                
        # Normalisation [0, 1]
        p_min, p_max = perlin_map.min(), perlin_map.max()
        perlin_map = (perlin_map - p_min) / (p_max - p_min + 1e-8)
        
        # Sauvegarde compressée .npy
        mask_path = os.path.join(output_dir, f"perlin_mask_{k:04d}.npy")
        np.save(mask_path, perlin_map.astype(np.float32))

    print(f"🎉 {num_masks} masques de bruit de Perlin générés avec succès dans : {output_dir}")

if __name__ == "__main__":
    generate_perlin_masks()
