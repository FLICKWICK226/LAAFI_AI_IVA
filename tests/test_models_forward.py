import pytest
import torch
from src.models.student_model import MobileNetV4Student

def test_student_mobilenetv4_forward():
    """Vérifie la passe forward et les cartes de caractéristiques du Student MobileNetV4."""
    model = MobileNetV4Student(pretrained=False, num_classes=1)
    dummy_input = torch.randn(2, 3, 384, 384)

    logits = model(dummy_input)
    assert logits.shape == (2, 1), f"Shape inattendue des logits : {logits.shape}"

    feat_map = model.get_last_feature_map()
    assert feat_map is not None, "La feature map intermédiaire est None."
    assert feat_map.shape[0] == 2, "Batch size incorrect sur la feature map."
    assert feat_map.ndim == 4, "La feature map doit être en 4D [B, C, H, W]."
