"""
Backward compatibility layer for cervix_transforms.
Imports SpecularReflectionMasker directly from the deep CervicalImagePipeline module.
"""

from src.preprocessing.cervix_pipeline import SpecularReflectionMasker, CervicalImagePipeline

__all__ = ["SpecularReflectionMasker", "CervicalImagePipeline"]
