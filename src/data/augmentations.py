import os
import cv2
import numpy as np
import albumentations as A

class FastPerlinNoiseLoader:
    """
    Chargeur haute performance de masques de bruit de Perlin pré-générés.
    Simule la présence de sang et de glaire par Alpha Blending sans ralentissement CPU.
    """
    def __init__(self, masks_dir: str = "./data/synthetic_masks"):
        self.masks_dir = masks_dir
        self.mask_files = []
        if os.path.exists(masks_dir):
            self.mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.npy')]

    def add_blood_or_mucus(self, image: np.ndarray, noise_type: str = 'blood', max_alpha: float = 0.4) -> np.ndarray:
        if not self.mask_files:
            return image  # Fallback si les masques ne sont pas encore disponibles
            
        h, w, c = image.shape
        mask_path = np.random.choice(self.mask_files)
        perlin = np.load(mask_path)
        
        if perlin.shape != (h, w):
            perlin = cv2.resize(perlin, (w, h), interpolation=cv2.INTER_LINEAR)
            
        mask = (perlin > 0.6).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        overlay = image.copy()
        if noise_type == 'blood':
            overlay[:, :, 0] = np.random.randint(10, 30)   # B
            overlay[:, :, 1] = np.random.randint(10, 30)   # G
            overlay[:, :, 2] = np.random.randint(130, 180) # R
        elif noise_type == 'mucus':
            overlay[:, :, 0] = np.random.randint(160, 190) # B
            overlay[:, :, 1] = np.random.randint(200, 220) # G
            overlay[:, :, 2] = np.random.randint(210, 230) # R

        alpha = mask[:, :, np.newaxis] * max_alpha
        blended = (image * (1 - alpha) + overlay * alpha).astype(np.uint8)
        return blended

def build_iva_augmentation_pipeline(is_train: bool = True) -> A.Compose:
    """
    Pipeline d'augmentation robuste pour imagerie de terrain IVA.
    RESPECTE LA RÈGLE 3 : Hue shift <= 0.05 pour ne pas altérer la réaction acéto-blanche.
    """
    if not is_train:
        return A.Compose([
            A.Resize(384, 384),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    return A.Compose([
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.4), p=0.5),
            A.MotionBlur(blur_limit=(3, 9), p=0.5),
        ], p=0.5),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.05, p=0.5), # Hue bridé à 0.05
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.3), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.4), p=0.5),
        ], p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
