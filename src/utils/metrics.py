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

def evaluate_threshold_grid(y_true: np.ndarray, y_prob: np.ndarray, min_t: float = 0.25, max_t: float = 0.45, step: float = 0.01):
    """
    Pilier 1 (Action 1.1) : Balayage fin du seuil T dans [0.25, 0.45] par pas de 0.01
    pour trouver le compromis optimal (Sensibilité >= 95.0% et Spécificité maximale > 80.0%).
    """
    grid_results = []
    best_item = None
    target_recall = 0.95

    thresholds = np.arange(min_t, max_t + step / 2.0, step)
    for t in thresholds:
        m = calculate_clinical_metrics(y_true, y_prob, threshold=float(t))
        grid_results.append(m)
        
        # Sélection du meilleur point : Sensibilité >= 95% avec Spécificité maximale
        if m['sensitivity'] >= target_recall:
            if best_item is None or m['specificity'] > best_item['specificity']:
                best_item = m

    if best_item is None and len(grid_results) > 0:
        # Fallback sur le meilleur F2 score
        best_item = max(grid_results, key=lambda x: x['f2_score'])

    return {
        "grid": grid_results,
        "optimal": best_item
    }

def categorize_tri_class(y_prob: np.ndarray, low_t: float = 0.20, high_t: float = 0.38) -> np.ndarray:
    """
    Pilier 1 (Action 1.2) : Moteur de triage tri-classe clinique :
    - 0 : GREEN (P < 0.20) -> Négatif (Contrôle 3 ans)
    - 1 : YELLOW (0.20 <= P < 0.38) -> Incertain (2nd badigeon acide ou 2e avis)
    - 2 : RED (P >= 0.38) -> Positif (Référer / Traitement)
    """
    categories = np.zeros(len(y_prob), dtype=int)
    categories[(y_prob >= low_t) & (y_prob < high_t)] = 1  # Yellow
    categories[y_prob >= high_t] = 2                      # Red
    return categories

