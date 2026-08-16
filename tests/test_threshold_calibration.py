import pytest
import numpy as np
from src.utils.metrics import calculate_clinical_metrics, evaluate_threshold_grid, categorize_tri_class

def test_calculate_clinical_metrics_exactness():
    """Vérifie l'exactitude mathématique du Recall, de la Spécificité et du Score F2."""
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.1, 0.2, 0.3, 0.6]) # TP=3, FN=1, TN=3, FP=1 à T=0.5

    metrics = calculate_clinical_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["tp"] == 3
    assert metrics["fn"] == 1
    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert pytest.approx(metrics["sensitivity"], 0.01) == 0.75
    assert pytest.approx(metrics["specificity"], 0.01) == 0.75
    assert metrics["f2_score"] > 0.0

def test_evaluate_threshold_grid_satisfies_target_recall():
    """Vérifie que l'algorithme sélectionne toujours un seuil respectant la sensibilité cible (>= 95%)."""
    np.random.seed(42)
    # Simulation d'un modèle bien entraîné
    y_true = np.random.choice([0, 1], size=200, p=[0.7, 0.3])
    y_prob = np.where(y_true == 1, np.random.beta(6, 2, size=200), np.random.beta(2, 6, size=200))

    results = evaluate_threshold_grid(y_true, y_prob, min_t=0.10, max_t=0.90, step=0.01)
    optimal = results["optimal"]

    assert optimal is not None, "Aucun point optimal trouvé."
    assert optimal["sensitivity"] >= 0.95, f"La sensibilité {optimal['sensitivity']} est inférieure au seuil de sécurité 95%."

def test_categorize_tri_class_labels():
    """Vérifie la bonne affectation des classes Vert (0), Jaune (1), Rouge (2)."""
    probs = np.array([0.05, 0.25, 0.50])
    categories = categorize_tri_class(probs, low_t=0.20, high_t=0.38)

    assert categories[0] == 0, "Doit être Vert (P < 0.20)"
    assert categories[1] == 1, "Doit être Jaune (0.20 <= P < 0.38)"
    assert categories[2] == 2, "Doit être Rouge (P >= 0.38)"
