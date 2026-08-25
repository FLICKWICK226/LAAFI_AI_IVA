import os
import cv2
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
import pytest
from PIL import Image

from src.data.dataset import IVADataset
from src.data.augmentations import FastPerlinNoiseLoader

def test_iva_dataset_loading_and_shapes():
    """Vérifie le chargement, les formes de tenseurs, les labels et l'indexation de IVADataset."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Création de 6 fausses images
        image_paths = []
        labels = ["Type_1", "Type_2", "Type_3", "Type_1", "Type_2", "Type_3"]
        for i, lbl in enumerate(labels):
            p = os.path.join(temp_dir, f"img_{i}.jpg")
            img = np.full((300, 300, 3), (i * 40) % 256, dtype=np.uint8)
            Image.fromarray(img).save(p)
            image_paths.append(p)

        csv_path = os.path.join(temp_dir, "dataset.csv")
        df = pd.DataFrame({
            "filepath": image_paths,
            "label": labels,
            "patient_id": [f"patient_{i // 2}" for i in range(len(labels))]
        })
        df.to_csv(csv_path, index=False)

        # 1. Dataset en mode Entraînement (avec Perlin proba)
        ds_train = IVADataset(csv_file=csv_path, is_train=True, masks_dir=temp_dir, perlin_proba=0.5)
        assert len(ds_train) == 6

        img_tensor, target, patient_id = ds_train[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 224, 224), f"Shape attendue [3, 224, 224], obtenu {img_tensor.shape}"
        assert target.item() == 0 # Type_1 -> 0
        assert target.dtype == torch.long
        assert patient_id == "patient_0"

        # 2. Dataset en mode Validation
        ds_val = IVADataset(csv_file=csv_path, is_train=False)
        img_tensor_val, target_val, p_val = ds_val[2]
        assert img_tensor_val.shape == (3, 224, 224)
        assert target_val.item() == 2 # Type_3 -> 2
        assert p_val == "patient_1"

    finally:
        shutil.rmtree(temp_dir)

def test_iva_dataset_missing_file_fallback():
    """Vérifie la robustesse en cas d'image manquante ou corrompue (retourne un tenseur noir valide)."""
    temp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(temp_dir, "corrupt.csv")
        df = pd.DataFrame({
            "filepath": [os.path.join(temp_dir, "non_existent.jpg")],
            "label": ["Type_2"],
            "patient_id": ["pat_x"]
        })
        df.to_csv(csv_path, index=False)

        ds = IVADataset(csv_file=csv_path, is_train=False)
        img_tensor, target, p_id = ds[0]

        assert img_tensor.shape == (3, 224, 224)
        assert target.item() == 1
        assert p_id == "pat_x"
    finally:
        shutil.rmtree(temp_dir)
