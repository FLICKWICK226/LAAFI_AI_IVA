"""
LAAFI_AI_IVA - Deep Cervical Image Pipeline Module
Consolidates image ingestion, specular reflection suppression, Perlin noise artifact injection,
WHO-compliant spatial/color augmentations (Hue <= 0.05), and ImageNet normalization behind a unified interface.
"""

import os
import cv2
import numpy as np
import torch
import albumentations as A
from typing import Union, Tuple, Optional, List

class SpecularReflectionMasker:
    """
    Suppression ultra-rapide (<2ms CPU) des reflets spéculaires intenses du flash LED (V > v_threshold en HSV).
    Remplace les pixels saturés par la couleur moyenne de la muqueuse non saturée.
    """
    def __init__(self, v_threshold: int = 235):
        self.v_threshold = v_threshold

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if not isinstance(img_np, np.ndarray) or img_np.ndim != 3:
            return img_np

        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        _, mask = cv2.threshold(v_channel, self.v_threshold, 255, cv2.THRESH_BINARY)
        
        if np.any(mask == 255):
            non_sat_mask = cv2.bitwise_not(mask)
            mean_color = cv2.mean(img_np, mask=non_sat_mask)[:3]
            img_np = img_np.copy()
            img_np[mask == 255] = np.array(mean_color, dtype=np.uint8)
            
        return img_np

class FastPerlinNoiseLoader:
    """
    Chargeur et générateur de masques de bruit de Perlin pour simuler les artefacts de mucus/sang.
    Pré-charge les masques en mémoire vive pour des augmentations à 0 ms d'E/S disque.
    """
    def __init__(
        self,
        masks_dir: str = "./data/synthetic_masks",
        target_size: Tuple[int, int] = (224, 224),
        max_cached_masks: int = 1000
    ):
        # Résolution dynamique des chemins selon l'environnement
        if os.path.exists("/content/data_fast/synthetic_masks"):
            masks_dir = "/content/data_fast/synthetic_masks"
        elif os.path.exists("/kaggle/working/data/synthetic_masks"):
            masks_dir = "/kaggle/working/data/synthetic_masks"

        self.masks_dir = masks_dir
        self.target_size = target_size
        self.masks_cache: List[np.ndarray] = []

        if os.path.exists(masks_dir):
            mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir) if f.endswith('.npy')][:max_cached_masks]
            for mf in mask_files:
                try:
                    perlin = np.load(mf).astype(np.float32)
                    mask = (perlin > 0.6).astype(np.float32)
                    mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
                    if mask_blurred.shape[:2] != target_size:
                        mask_blurred = cv2.resize(mask_blurred, target_size, interpolation=cv2.INTER_AREA)
                    self.masks_cache.append(mask_blurred[:, :, np.newaxis].astype(np.float32))
                except Exception:
                    pass

        # Fallback procédural si aucun masque sur disque
        if len(self.masks_cache) == 0:
            for _ in range(5):
                raw_noise = np.random.uniform(0, 1, target_size).astype(np.float32)
                mask = (raw_noise > 0.65).astype(np.float32)
                mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)[:, :, np.newaxis]
                self.masks_cache.append(mask_blurred)

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

class CervicalImagePipeline:
    """
    Deep Module pour le traitement complet de l'imagerie IVA cervicale :
    - Ingestion robuste (chemin de fichier ou array NumPy)
    - Masquage des reflets spéculaires
    - Injection contrôlée d'artefacts (Perlin blood/mucus)
    - Augmentations géométriques & colorimétriques conformes OMS (Hue <= 0.05)
    - Normalisation ImageNet et conversion en torch.Tensor [3, H, W]
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        specular_threshold: int = 235,
        perlin_proba: float = 0.30,
        masks_dir: str = "./data/synthetic_masks",
        strict: bool = False
    ):
        self.img_size = img_size
        self.perlin_proba = perlin_proba
        self.strict = strict
        self.specular_masker = SpecularReflectionMasker(v_threshold=specular_threshold)
        self.perlin_loader = FastPerlinNoiseLoader(masks_dir=masks_dir, target_size=img_size)

        h, w = img_size
        self.train_transform = A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.4),
            A.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.05, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.val_transform = A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_image(self, image_input: Union[str, np.ndarray], is_train: bool = True) -> np.ndarray:
        """
        Décode l'image depuis un fichier ou valide un tableau NumPy.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                if self.strict and not is_train:
                    raise FileNotFoundError(f"Image introuvable : {image_input}")
                # Fallback neutre sécurisé
                return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

            img = cv2.imread(image_input)
            if img is None:
                if self.strict and not is_train:
                    raise ValueError(f"Fichier image corrompu : {image_input}")
                return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb

        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.ndim == 3 and image_input.shape[2] == 3:
                img_rgb = image_input
            else:
                raise ValueError(f"Shape image invalide : {image_input.shape}")
            return img_rgb
        else:
            raise TypeError(f"Type d'entrée image non supporté : {type(image_input)}")

    def process(self, image_input: Union[str, np.ndarray], is_train: bool = True) -> torch.Tensor:
        """
        Traite une image cervicale et retourne un tenseur PyTorch standardisé [3, H, W].
        """
        # 1. Ingestion de l'image
        img = self.load_image(image_input, is_train=is_train)

        # 2. Prétraitements spécifiques IVA
        if is_train:
            # Suppression des reflets spéculaires
            img = self.specular_masker(img)

            # Injection d'artefacts synthétiques de Perlin (mucus / sang)
            if np.random.rand() < self.perlin_proba:
                noise_type = 'blood' if np.random.rand() < 0.5 else 'mucus'
                img = self.perlin_loader.add_blood_or_mucus(img, noise_type=noise_type)

            # Augmentations spatiales et colorimétriques
            augmented = self.train_transform(image=img)['image']
        else:
            # Mode inférence / validation (déterministe)
            augmented = self.val_transform(image=img)['image']

        # 3. Conversion en tenseur PyTorch [3, H, W]
        tensor = torch.from_numpy(augmented.transpose(2, 0, 1)).float()
        return tensor

    def __call__(self, image_input: Union[str, np.ndarray], is_train: bool = True) -> torch.Tensor:
        return self.process(image_input, is_train=is_train)
