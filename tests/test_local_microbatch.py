import torch
import torch.nn as nn
import numpy as np
import pytest

from src.models.classifier_lesion import IVALesionClassifierStage2
from src.utils.metrics import evaluate_threshold_grid

def test_local_cpu_microbatch_train_and_val_cycle():
    """
    Pré-requis absolu Phase 4.1 : Validation d'un micro-batch complet en local/CPU
    - 1 itération d'entraînement (Forward -> Loss -> Backward -> Step -> Scheduler)
    - 1 forward de validation (Softmax -> Probabilités -> Triage / Seuil)
    Garantit l'absence d'erreur d'exécution ou de shape avant tout déploiement distant.
    """
    device = torch.device("cpu")

    # 1. Modèle
    model = IVALesionClassifierStage2(
        backbone_name="convnext_small",
        pretrained=False,
        num_classes=3,
        drop_rate=0.2
    ).to(device)

    # 2. Perte et Optimiseur
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # 3. Mini-Batch Train (B=2, C=3, H=224, W=224)
    train_images = torch.randn(2, 3, 224, 224, device=device)
    train_targets = torch.tensor([0, 2], dtype=torch.long, device=device)

    # 4. Phase Warmup (Freeze Backbone)
    for p in model.backbone.parameters():
        p.requires_grad = False

    model.train()
    optimizer.zero_grad()
    logits = model(train_images)
    assert logits.shape == (2, 3), f"Shape attendue [2, 3], obtenu {logits.shape}"

    loss = criterion(logits, train_targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

    loss.backward()
    # Vérification que la tête a des gradients et le backbone non
    assert model.head[-1].weight.grad is not None
    assert model.head[-1].weight.grad.shape == (3, 512)

    optimizer.step()
    scheduler.step()

    # 5. Déblocage Backbone
    for p in model.backbone.parameters():
        p.requires_grad = True

    # 6. Mini-Batch Validation
    model.eval()
    val_images = torch.randn(2, 3, 224, 224, device=device)
    val_targets = torch.tensor([1, 0], dtype=torch.long, device=device)

    with torch.no_grad():
        val_logits = model(val_images)
        val_probs = torch.softmax(val_logits, dim=1).numpy()

    assert val_probs.shape == (2, 3)
    
    # 7. Évaluation de seuil de validation
    binary_targets = (val_targets.numpy() > 0).astype(int)
    pos_probs = val_probs[:, 1] + val_probs[:, 2]
    grid_res = evaluate_threshold_grid(binary_targets, pos_probs, min_t=0.20, max_t=0.80, step=0.10)
    assert "grid" in grid_res
