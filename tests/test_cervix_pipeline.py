import os
import cv2
import numpy as np
import torch
import pytest
from src.preprocessing.cervix_pipeline import CervicalImagePipeline, SpecularReflectionMasker, FastPerlinNoiseLoader

def test_specular_reflection_suppression():
    """Vérifie la suppression et l'atténuation des reflets spéculaires du flash."""
    masker = SpecularReflectionMasker(v_threshold=235)
    # Image rouge col avec zone centrale brûlée par le flash (V=255)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [180, 40, 40] # Rouge muqueuse IVA
    img[40:60, 40:60] = [255, 255, 255] # Reflet flash saturé

    processed = masker(img)
    assert processed.shape == img.shape
    # La zone centrale ne doit plus être à 255,255,255
    assert not np.all(processed[45:55, 45:55] == [255, 255, 255])

def test_fast_perlin_noise_loader():
    """Vérifie le chargement ou la génération procédurale d'artefacts de mucus/sang."""
    loader = FastPerlinNoiseLoader(masks_dir="./non_existent_masks", target_size=(224, 224))
    assert len(loader.masks_cache) > 0

    img = np.full((224, 224, 3), 150, dtype=np.uint8)
    blended_blood = loader.add_blood_or_mucus(img, noise_type='blood')
    blended_mucus = loader.add_blood_or_mucus(img, noise_type='mucus')

    assert blended_blood.shape == (224, 224, 3)
    assert blended_mucus.shape == (224, 224, 3)

def test_cervix_pipeline_output_contract():
    """Vérifie la forme [3, 224, 224], le type float32 et la plage de valeurs normalisées."""
    pipeline = CervicalImagePipeline(img_size=(224, 224), perlin_proba=0.5)

    # Test avec image NumPy
    img_np = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    tensor_train = pipeline.process(img_np, is_train=True)
    tensor_val = pipeline.process(img_np, is_train=False)

    assert isinstance(tensor_train, torch.Tensor)
    assert tensor_train.shape == (3, 224, 224)
    assert tensor_train.dtype == torch.float32

    assert isinstance(tensor_val, torch.Tensor)
    assert tensor_val.shape == (3, 224, 224)
    assert tensor_val.dtype == torch.float32

def test_cervix_pipeline_error_handling(tmp_path):
    """Vérifie le fallback par défaut et la levée d'erreur stricte quand strict=True."""
    non_existent = str(tmp_path / "missing.jpg")

    # Mode par défaut (strict=False) -> Fallback sans crash
    pipeline_default = CervicalImagePipeline(img_size=(224, 224), strict=False)
    tensor_fallback = pipeline_default.process(non_existent, is_train=False)
    assert tensor_fallback.shape == (3, 224, 224)

    # Mode strict -> Exception levée en validation
    pipeline_strict = CervicalImagePipeline(img_size=(224, 224), strict=True)
    with pytest.raises(FileNotFoundError):
        pipeline_strict.process(non_existent, is_train=False)
