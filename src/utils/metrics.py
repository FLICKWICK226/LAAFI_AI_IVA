import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, fbeta_score

class FocalLoss(nn.Module):
    """
    Focal Loss pour équilibrer les classes et fortement pénaliser les Faux Négatifs (Rule 5).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def calculate_clinical_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """
    Calcule les métriques de sécurité clinique imposées par la PRD v2.0 :
    - Sensibilité (Recall) >= 95%
    - Spécificité >= 80%
    - Score F2 >= 0.88
    - AUC-ROC >= 0.90
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    # Matrice de confusion
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    
    # Score F2 (pondère le Recall 2x plus que la Précision)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_roc = 0.5

    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f2_score": float(f2),
        "auc_roc": float(auc_roc),
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }
