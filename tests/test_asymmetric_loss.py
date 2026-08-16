import pytest
import torch
from src.losses.asymmetric_loss import AsymmetricFocalLoss

def test_asymmetric_focal_loss_forward():
    """Vérifie le calcul et la dimension de sortie scalaire d'AsymmetricFocalLoss."""
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0)
    logits = torch.randn(4, 1, requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

    loss = criterion(logits, targets)
    assert loss.ndim == 0, "La perte doit être un scalaire."
    assert not torch.isnan(loss), "La perte contient des NaN."
    assert not torch.isinf(loss), "La perte contient des Inf."
    assert loss.item() >= 0.0, "La perte doit être positive ou nulle."

def test_asymmetric_focal_loss_backward():
    """Vérifie que la perte rétropropage correctement les gradients sans explosion."""
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0)
    logits = torch.randn(4, 1, requires_grad=True)
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])

    loss = criterion(logits, targets)
    loss.backward()

    assert logits.grad is not None, "Le gradient n'a pas été calculé."
    assert not torch.isnan(logits.grad).any(), "Gradients NaN détectés."

def test_asymmetric_focal_loss_two_classes():
    """Vérifie la compatibilité avec un tenseur de logits à 2 classes [B, 2]."""
    criterion = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0)
    logits = torch.randn(4, 2, requires_grad=True)
    targets = torch.tensor([0, 1, 0, 1])

    loss = criterion(logits, targets)
    assert loss.ndim == 0
    assert loss.item() >= 0.0
