import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import os

@pytest.fixture(autouse=True)
def set_random_seed():
    """Garantit la reproductibilité absolue pour chaque test unitaire."""
    torch.manual_seed(42)
    np.random.seed(42)

@pytest.fixture
def synthetic_batch():
    """Fournit un mini-batch synthétique d'images (B=2, C=3, H=384, W=384) et cibles binaires."""
    images = torch.randn(2, 3, 384, 384, dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    return images, targets

@pytest.fixture
def dummy_patient_dataframe():
    """Fournit un DataFrame synthétique simulant plusieurs clichés par patiente."""
    data = {
        "image_path": [f"img_{i}.jpg" for i in range(20)],
        "patient_id": [f"patient_{i // 4}" for i in range(20)], # 5 patientes, 4 clichés chacune
        "label": [i % 2 for i in range(20)]
    }
    return pd.DataFrame(data)
