import os
import tempfile
import shutil
import numpy as np
import pytest

from src.data.augmentations import FastPerlinNoiseLoader, build_iva_augmentation_pipeline
from src.preprocessing.cervix_transforms import SpecularReflectionMasker

def test_fast_perlin_noise_loader():
    """Vérifie le chargement et l'application des masques de bruit de Perlin (sang et glaire)."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Création de 3 faux masques .npy (valeurs float [0, 1])
        for i in range(3):
            mask = np.random.rand(100, 100).astype(np.float32)
            np.save(os.path.join(temp_dir, f"perlin_{i}.npy"), mask)

        loader = FastPerlinNoiseLoader(masks_dir=temp_dir, max_cached_masks=10)
        assert len(loader.masks_cache) == 3

        img = np.full((200, 200, 3), 120, dtype=np.uint8)

        # Test injection de sang
        blood_blended = loader.add_blood_or_mucus(img, noise_type="blood", max_alpha=0.4)
        assert blood_blended.shape == (200, 200, 3)
        assert blood_blended.dtype == np.uint8

        # Test injection de mucus
        mucus_blended = loader.add_blood_or_mucus(img, noise_type="mucus", max_alpha=0.4)
        assert mucus_blended.shape == (200, 200, 3)
        assert mucus_blended.dtype == np.uint8
    finally:
        shutil.rmtree(temp_dir)

def test_augmentation_pipeline_train_vs_val():
    """Vérifie les différences d'augmentation entre mode entraînement et validation."""
    pipe_train = build_iva_augmentation_pipeline(is_train=True, img_size=(224, 224))
    pipe_val = build_iva_augmentation_pipeline(is_train=False, img_size=(224, 224))

    dummy_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)

    # Mode Train
    aug_train = pipe_train(image=dummy_img)["image"]
    assert aug_train.shape == (224, 224, 3)
    assert aug_train.dtype == np.float32 # Normalisé ImageNet

    # Mode Validation (Déterministe)
    aug_val1 = pipe_val(image=dummy_img)["image"]
    aug_val2 = pipe_val(image=dummy_img)["image"]
    assert aug_val1.shape == (224, 224, 3)
    np.testing.assert_array_almost_equal(aug_val1, aug_val2, decimal=5)
