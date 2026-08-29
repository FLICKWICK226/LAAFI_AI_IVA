"""
Augmentations and image processing interface.
Aliases FastPerlinNoiseLoader and build_iva_augmentation_pipeline to the deep CervicalImagePipeline module.
"""

from src.preprocessing.cervix_pipeline import (
    FastPerlinNoiseLoader,
    SpecularReflectionMasker,
    CervicalImagePipeline
)
import albumentations as A

def build_iva_augmentation_pipeline(
    is_train: bool = True,
    img_size: tuple = (224, 224),
    specular_proba: float = 0.3
) -> A.Compose:
    """
    Pipeline Albumentations conforme aux règles cliniques OMS (Hue Shift <= 0.05).
    """
    pipeline = CervicalImagePipeline(img_size=img_size)
    return pipeline.train_transform if is_train else pipeline.val_transform

__all__ = [
    "FastPerlinNoiseLoader",
    "SpecularReflectionMasker",
    "CervicalImagePipeline",
    "build_iva_augmentation_pipeline"
]
