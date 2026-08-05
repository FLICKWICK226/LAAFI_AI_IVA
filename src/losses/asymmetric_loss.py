import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss compatible avec logits uniques [B, 1] ou 2-classes [B, 2].
    Désactive dynamiquement le gradient des négatifs faciles pour augmenter la Spécificité
    sans détruire la Sensibilité (Recall).
    """
    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float().view(-1)

        if logits.ndim > 1 and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
        elif logits.ndim > 1 and logits.shape[1] == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
        else:
            probs = torch.sigmoid(logits.view(-1))

        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        probs_neg = (1.0 - probs).clamp(min=self.eps, max=1.0 - self.eps)

        if self.clip > 0:
            probs_neg_clipped = (probs_neg + self.clip).clamp(max=1.0)
        else:
            probs_neg_clipped = probs_neg

        loss_pos = targets * torch.pow(1.0 - probs_pos, self.gamma_pos) * torch.log(probs_pos)
        loss_neg = (1.0 - targets) * torch.pow(1.0 - probs_neg_clipped, self.gamma_neg) * torch.log(probs_neg)

        loss = - (loss_pos + loss_neg)
        return loss.mean()
