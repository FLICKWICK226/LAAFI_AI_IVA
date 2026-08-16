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

def evaluate_threshold_grid(y_true: np.ndarray, y_prob: np.ndarray, min_t: float = 0.10, max_t: float = 0.90, step: float = 0.01):
    """
    Pilier 1 (Action 1.1) : Balayage fin du seuil T dans [0.10, 0.90] par pas de 0.01
    pour trouver le compromis optimal (Sensibilité >= 95.0% et Spécificité maximale >= 80.0%).
    Évite les angles morts dus au décalage de probabilité de la FocalLoss (alpha=0.75).
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


def calculate_anatomical_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    """
    Calcule les métriques rigoureuses de classification anatomique (Type 1, Type 2, Type 3) :
    - Accuracy globale
    - Macro-F1 et Weighted-F1
    - Précision et Rappel par classe
    - Matrice de confusion 3x3
    - Macro AUC-ROC One-vs-Rest
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    try:
        auc_roc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr', average='macro')
    except Exception:
        auc_roc = 0.5

    # Rappel et précision par classe
    report = classification_report(y_true, y_pred, target_names=['Type_1', 'Type_2', 'Type_3'], output_dict=True, zero_division=0)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_auc_roc": float(auc_roc),
        "confusion_matrix": cm,
        "type_1": report.get("Type_1", {}),
        "type_2": report.get("Type_2", {}),
        "type_3": report.get("Type_3", {}),
    }


def calculate_clinical_triage_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray, referral_threshold: float = 0.35) -> dict:
    """
    Moteur de Triage Clinique SaMD (Directives OMS / IFCPC Screen-and-Treat) :
    - ÉLIGIBLE TRAITEMENT LOCAL (Type 1 + Type 2) -> Label 1
    - RÉFÉRENCE CHIRURGICALE CHU (Type 3)         -> Label 0

    Sécurité Patient : Si P(Type 3) >= referral_threshold (défaut 0.35),
    la patiente est référée au centre expert (Feu Rouge cryothérapie).
    """
    from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, confusion_matrix

    # Vrai statut clinique : 1 = Éligible (Types 1 et 2), 0 = Inéligible (Type 3)
    y_true_eligible = (y_true != 2).astype(int)

    # Probabilité d'être éligible et d'être référé
    prob_eligible = y_pred_probs[:, 0] + y_pred_probs[:, 1]
    prob_referral = y_pred_probs[:, 2]

    # Décision de triage clinique (Sécurisée)
    # Si le risque de Type 3 dépasse le seuil, la patiente est référée (y_pred_eligible = 0)
    y_pred_eligible = (prob_referral < referral_threshold).astype(int)

    # Métriques cliniques de triage
    triage_acc = accuracy_score(y_true_eligible, y_pred_eligible)
    sens_eligible = recall_score(y_true_eligible, y_pred_eligible, pos_label=1, zero_division=0)
    spec_safety = recall_score(y_true_eligible, y_pred_eligible, pos_label=0, zero_division=0) # Taux de Type 3 bien référés
    prec_eligible = precision_score(y_true_eligible, y_pred_eligible, pos_label=1, zero_division=0)

    try:
        triage_auc = roc_auc_score(y_true_eligible, prob_eligible)
    except Exception:
        triage_auc = 0.5

    cm = confusion_matrix(y_true_eligible, y_pred_eligible, labels=[0, 1]).tolist()

    return {
        "triage_accuracy": float(triage_acc),
        "sensitivity_eligible": float(sens_eligible),
        "safety_specificity_type3": float(spec_safety),
        "precision_eligible": float(prec_eligible),
        "triage_auc_roc": float(triage_auc),
        "referral_threshold": float(referral_threshold),
        "confusion_matrix_2x2": {
            "true_referred_type3": cm[0][0],
            "false_eligible_risk": cm[0][1],
            "false_referred_type1_2": cm[1][0],
            "true_eligible": cm[1][1]
        }
    }


