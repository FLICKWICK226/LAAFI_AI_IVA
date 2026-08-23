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

def test_stage2_classifier_forward():
    """Vérifie la passe forward du classifieur Stage 2 unifié à 3 classes en résolution 224x224."""
    from src.models.classifier_lesion import IVALesionClassifierStage2
    model = IVALesionClassifierStage2(backbone_name="convnext_small", pretrained=False, num_classes=3)
    dummy_input = torch.randn(2, 3, 224, 224)

    outputs = model(dummy_input)
    assert isinstance(outputs, torch.Tensor), f"Type inattendu : {type(outputs)}"
    assert outputs.shape == (2, 3), f"Shape logits inattendue : {outputs.shape} (attendu [2, 3])"

def test_anatomical_and_triage_metrics():
    """Vérifie le bon calcul des métriques tri-classes et du moteur de triage SaMD."""
    import numpy as np
    from src.utils.metrics import calculate_anatomical_metrics, calculate_clinical_triage_metrics

    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_probs = np.array([
        [0.8, 0.15, 0.05], # Type 1 -> Éligible
        [0.1, 0.8, 0.1],   # Type 2 -> Éligible
        [0.05, 0.15, 0.8], # Type 3 -> Référé
        [0.7, 0.2, 0.1],   # Type 1 -> Éligible
        [0.15, 0.75, 0.1], # Type 2 -> Éligible
        [0.02, 0.08, 0.9], # Type 3 -> Référé
    ])

    anat = calculate_anatomical_metrics(y_true, y_probs)
    assert anat["accuracy"] == 1.0
    assert anat["macro_f1"] == 1.0

    triage = calculate_clinical_triage_metrics(y_true, y_probs, referral_threshold=0.35)
    assert triage["triage_accuracy"] == 1.0
    assert triage["sensitivity_eligible"] == 1.0
    assert triage["safety_specificity_type3"] == 1.0

def test_export_onnx_single_head(tmp_path):
    """Vérifie que l'exportation ONNX du modèle unifié fonctionne sans erreur."""
    import os
    import onnx
    from src.models.export_onnx import export_model_to_onnx

    output_onnx = str(tmp_path / "test_model.onnx")
    export_model_to_onnx(
        checkpoint_path=None,
        output_onnx_path=output_onnx,
        img_size=(224, 224),
        backbone_name="convnext_small"
    )

    assert os.path.exists(output_onnx), "Le fichier ONNX n'a pas été créé."
    onnx_proto = onnx.load(output_onnx)
    onnx.checker.check_model(onnx_proto)
    assert len(onnx_proto.graph.output) == 1, "Le modèle doit avoir exactement 1 sortie."
    assert onnx_proto.graph.output[0].name == "logits"

