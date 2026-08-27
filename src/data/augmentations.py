import os
import cv2
import numpy as np
import albumentations as A
from src.preprocessing.cervix_transforms import SpecularReflectionMasker

class FastPerlinNoiseLoader:
    """
    Chargeur ultra-optimisé de masques de bruit de Perlin.
    Les masques sont floutés, redimensionnés en (224, 224) et pré-chargés en RAM pour des augmentations à 0 ms d'E/S.
    """
    def __init__(self, masks_dir: str = "./data/synthetic_masks", target_size: tuple = (224, 224), max_cached_masks: int = 1000):
        # Résolution automatique SSD local si disponible sous Colab ou Kaggle
        if os.path.exists("/content/data_fast/synthetic_masks"):
            masks_dir = "/content/data_fast/synthetic_masks"
        elif os.path.exists("/kaggle/working/data/synthetic_masks"):
            masks_dir = "/kaggle/working/data/synthetic_masks"
            
        self.masks_dir = masks_dir
        self.target_size = target_size
        self.masks_cache = []
        
        if os.path.exists(masks_dir):
            mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.npy')][:max_cached_masks]
            if mask_files:
                for mf in mask_files:
                    try:
                        perlin = np.load(mf).astype(np.float32)
                        # Pré-calcul du flou gaussien et redimensionnement immédiat en RAM
                        mask = (perlin > 0.6).astype(np.float32)
                        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
                        if mask_blurred.shape[:2] != target_size:
                            mask_blurred = cv2.resize(mask_blurred, target_size, interpolation=cv2.INTER_AREA)
                        self.masks_cache.append(mask_blurred[:, :, np.newaxis].astype(np.float32))
                    except Exception:
                        pass
                if len(self.masks_cache) > 0:
                    print(f"🚀 {len(self.masks_cache)} masques de Perlin ({target_size[0]}x{target_size[1]}) mis en cache RAM.")

    def add_blood_or_mucus(self, image: np.ndarray, noise_type: str = 'blood', max_alpha: float = 0.4) -> np.ndarray:
        if not self.masks_cache:
            return image
            
        h, w, c = image.shape
        idx = np.random.randint(0, len(self.masks_cache))
        alpha = self.masks_cache[idx]
        
        if alpha.shape[:2] != (h, w):
            alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
            
        alpha_scaled = alpha * max_alpha
        overlay = np.empty_like(image)
        
        if noise_type == 'blood':
            overlay[:, :, 0] = np.random.randint(10, 30)   # B
            overlay[:, :, 1] = np.random.randint(10, 30)   # G
            overlay[:, :, 2] = np.random.randint(130, 180) # R
        elif noise_type == 'mucus':
            overlay[:, :, 0] = np.random.randint(160, 190) # B
            overlay[:, :, 1] = np.random.randint(200, 220) # G
            overlay[:, :, 2] = np.random.randint(210, 230) # R

        blended = (image * (1.0 - alpha_scaled) + overlay * alpha_scaled).astype(np.uint8)
        return blended

def build_iva_augmentation_pipeline(is_train: bool = True, img_size: tuple = (224, 224), specular_proba: float = 0.4) -> A.Compose:
    """
    Pipeline d'augmentation rapide et robuste pour imagerie de terrain IVA.
    Hue Shift strictly <= 0.05 per Rule 3.
    """
    h, w = img_size
    if not is_train:
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.4), p=0.5),
            A.MotionBlur(blur_limit=(3, 9), p=0.5),
        ], p=0.5),
        A.RandomSunFlare(
            flare_roi=(0.1, 0.1, 0.9, 0.9),
            p=specular_proba
        ),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.05, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

