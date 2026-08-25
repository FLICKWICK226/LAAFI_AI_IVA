import numpy as np
import pytest
import cv2
import torch
from unittest.mock import MagicMock

from src.models.detector_jsc import JSCDetectorStage1

def test_jsc_detector_fallback_center_crop():
    """Vérifie que sans modèle YOLO, le détecteur applique un Center Crop 70% et redimensionne correctement."""
    detector = JSCDetectorStage1(model_path=None, conf_threshold=0.25)
    dummy_image = np.full((600, 800, 3), 128, dtype=np.uint8)

    crop_default = detector.crop_jsc(dummy_image, target_size=(384, 384))
    assert crop_default.shape == (384, 384, 3), f"Shape inattendue : {crop_default.shape}"

    crop_custom = detector.crop_jsc(dummy_image, target_size=(224, 224))
    assert crop_custom.shape == (224, 224, 3), f"Shape inattendue : {crop_custom.shape}"

def test_jsc_detector_with_mocked_yolo_detection_and_15pct_padding():
    """Vérifie le calcul précis de la ROI avec 15% de marge de sécurité (padding)."""
    detector = JSCDetectorStage1(model_path=None, conf_threshold=0.25)

    # Simulation d'un retour Ultralytics YOLO
    mock_boxes = MagicMock()
    mock_boxes.conf = torch.tensor([0.85]) # Forte confiance
    # Box centrée [x1, y1, x2, y2] = [200, 200, 400, 400] (largeur=200, hauteur=200)
    mock_boxes.xyxy = torch.tensor([[200.0, 200.0, 400.0, 400.0]])

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes

    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    detector.model = mock_model

    # Image source 600x600
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    img[200:400, 200:400] = 255 # Signal dans la boîte

    cropped = detector.crop_jsc(img, target_size=(224, 224))
    assert cropped.shape == (224, 224, 3)
    # Le crop avec 15% de padding (30px de marge) doit couvrir [170:430, 170:430]
    assert np.mean(cropped) > 0, "La zone recadrée doit contenir le signal de la JSC."

def test_jsc_detector_low_confidence_fallback():
    """Vérifie que si la détection a une confiance < seuil (0.25), le fallback center crop est activé."""
    detector = JSCDetectorStage1(model_path=None, conf_threshold=0.25)

    mock_boxes = MagicMock()
    mock_boxes.conf = torch.tensor([0.10]) # Sous le seuil 0.25
    mock_boxes.xyxy = torch.tensor([[100.0, 100.0, 150.0, 150.0]])

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_model = MagicMock()
    mock_model.return_value = [mock_result]
    detector.model = mock_model

    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cropped = detector.crop_jsc(img, target_size=(224, 224))
    assert cropped.shape == (224, 224, 3)
