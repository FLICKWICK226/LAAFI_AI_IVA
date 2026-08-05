import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss pour classification binaire à logit unique [Batch_Size, 1].
    Désactive dynamiquement le gradient des négatifs faciles pour augmenter la Spécificité
    sans détruire la Sensibilité (Recall).
    
    FL_asymmetric = - y * (1 - p)^gamma_pos * log(p) - (1 - y) * max(p - clip, 0)^gamma_neg * log(1 - p)
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
        # Alignement des dimensions [B, 1] ou [B]
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        
        # Probabilité sigmoid p_i = P(y = 1 | x)
        probs = torch.sigmoid(logits)
        probs_pos = probs.clamp(min=self.eps, max=1.0 - self.eps)
        probs_neg = (1.0 - probs).clamp(min=self.eps, max=1.0 - self.eps)

        # Asymmetric Clipping pour supprimer le bruit des négatifs faciles
        if self.clip > 0:
            probs_neg_clipped = (probs_neg + self.clip).clamp(max=1.0)
        else:
            probs_neg_clipped = probs_neg

        # Calcul des pertes positives et négatives
        loss_pos = targets * torch.pow(1.0 - probs_pos, self.gamma_pos) * torch.log(probs_pos)
        loss_neg = (1.0 - targets) * torch.pow(1.0 - probs_neg_clipped, self.gamma_neg) * torch.log(probs_neg)

        loss = - (loss_pos + loss_neg)
        return loss.mean()
