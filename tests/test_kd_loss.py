import pytest
import torch
from src.distillation.kd_loss import BinaryHybridKDLoss

def test_binary_hybrid_kd_loss_forward():
    """Vérifie le forward de la perte de distillation hybride complète."""
    criterion = BinaryHybridKDLoss(alpha_kd=0.6, beta_attn=50.0, temperature=4.0)

    student_logits = torch.randn(2, 1, requires_grad=True)
    teacher_logits = torch.randn(2, 1)

    student_feats = torch.randn(2, 64, 12, 12, requires_grad=True)
    teacher_feats = torch.randn(2, 128, 12, 12)

    targets = torch.tensor([0, 1])

    loss = criterion(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_feats=student_feats,
        teacher_feats=teacher_feats,
        targets=targets
    )

    assert loss.ndim == 0, "La perte de distillation doit être un scalaire."
    assert not torch.isnan(loss), "La perte KD contient des NaN."
    assert loss.item() >= 0.0, "La perte KD doit être positive."

def test_binary_hybrid_kd_loss_backward():
    """Vérifie la rétropropagation des gradients sur les logits et features du student."""
    criterion = BinaryHybridKDLoss(alpha_kd=0.6, beta_attn=50.0, temperature=4.0)

    student_logits = torch.randn(2, 1, requires_grad=True)
    teacher_logits = torch.randn(2, 1)
    student_feats = torch.randn(2, 64, 12, 12, requires_grad=True)
    teacher_feats = torch.randn(2, 128, 12, 12)
    targets = torch.tensor([0, 1])

    loss = criterion(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_feats=student_feats,
        teacher_feats=teacher_feats,
        targets=targets
    )
    loss.backward()

    assert student_logits.grad is not None, "Gradient manquant sur student_logits."
    assert student_feats.grad is not None, "Gradient manquant sur student_feats."
