import os
import cv2
import tempfile
import shutil
import numpy as np
import torch
import pytest
from PIL import Image

from src.data.quality_filter import CervicalImageQualityFilter
from src.models.detector_jsc import JSCDetectorStage1
from src.models.classifier_lesion import IVALesionClassifierStage2
from src.utils.metrics import calculate_clinical_triage_metrics

def test_stage0_quality_gate_filtering():
    """Vérifie que le Stage 0 rejette correctement les images floues, sur-exposées ou trop sombres."""
    q_filter = CervicalImageQualityFilter(min_laplacian_var=40.0, max_overexposed_ratio=0.25, max_underexposed_ratio=0.35)
    temp_dir = tempfile.mkdtemp()
    try:
        # 1. Image Conforme (Texture nette et exposition équilibrée)
        sharp_img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
        p_sharp = os.path.join(temp_dir, "sharp.jpg")
        Image.fromarray(sharp_img).save(p_sharp)
        res_sharp = q_filter.evaluate_image(p_sharp)
        assert res_sharp["is_valid"] is True, f"Image nette rejetée : {res_sharp['rejection_reason']}"

        # 2. Image Floue (Laplacien < 40)
        blurry_img = cv2.GaussianBlur(sharp_img, (25, 25), 0)
        p_blur = os.path.join(temp_dir, "blurry.jpg")
        Image.fromarray(blurry_img).save(p_blur)
        res_blur = q_filter.evaluate_image(p_blur)
        assert res_blur["is_valid"] is False
        assert "BLURRY_IMAGE" in res_blur["rejection_reason"]

        # 3. Image Sur-exposée (Flash LED blanc > 25%)
        overexp_img = sharp_img.copy()
        overexp_img[:160, :] = 255
        p_over = os.path.join(temp_dir, "overexposed.jpg")
        Image.fromarray(overexp_img).save(p_over)
        res_over = q_filter.evaluate_image(p_over)
        assert res_over["is_valid"] is False
        assert "OVEREXPOSED_FLASH" in res_over["rejection_reason"]

        # 4. Image Sous-exposée (Trop sombre > 35%)
        underexp_img = np.zeros((300, 300, 3), dtype=np.uint8)
        p_under = os.path.join(temp_dir, "underexposed.jpg")
        Image.fromarray(underexp_img).save(p_under)
        res_under = q_filter.evaluate_image(p_under)
        assert res_under["is_valid"] is False
        assert "UNDEREXPOSED_DARK" in res_under["rejection_reason"]
    finally:
        shutil.rmtree(temp_dir)

def test_full_pipeline_stage0_to_stage2_e2e():
    """
    Test d'intégration complet Bout-en-Bout (Stage 0 -> Stage 1 -> Stage 2) :
    1. Stage 0 : Image brute validée par le Quality Gate.
    2. Stage 1 : Détection de la JSC et extraction du crop 224x224 avec marge 15%.
    3. Stage 2 : Passage dans le classifieur ConvNeXt-Small -> Logits [1, 3] -> Probabilités.
    4. Triage Clinique : Émission de la décision SaMD OMS.
    """
    # 1. Stage 0
    q_filter = CervicalImageQualityFilter()
    raw_img = np.random.randint(50, 200, (400, 400, 3), dtype=np.uint8)
    temp_dir = tempfile.mkdtemp()
    try:
        p_raw = os.path.join(temp_dir, "cervix_raw.jpg")
        Image.fromarray(raw_img).save(p_raw)
        q_res = q_filter.evaluate_image(p_raw)
        assert q_res["is_valid"] is True

        # 2. Stage 1
        detector = JSCDetectorStage1(model_path=None)
        crop = detector.crop_jsc(raw_img, target_size=(224, 224))
        assert crop.shape == (224, 224, 3)

        # 3. Normalisation & Tensorisation
        crop_tensor = torch.tensor(crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # 4. Stage 2
        model = IVALesionClassifierStage2(backbone_name="convnext_small", pretrained=False, num_classes=3)
        model.eval()
        with torch.no_grad():
            logits = model(crop_tensor)
            probs = torch.softmax(logits, dim=1).numpy() # [1, 3]

        assert logits.shape == (1, 3)
        assert pytest.approx(probs.sum(), 0.001) == 1.0

        # 5. Moteur de Triage Clinique OMS
        triage = calculate_clinical_triage_metrics(
            y_true=np.array([0]),
            y_pred_probs=probs,
            referral_threshold=0.35
        )
        assert "triage_accuracy" in triage
        assert "sensitivity_eligible" in triage
    finally:
        shutil.rmtree(temp_dir)
