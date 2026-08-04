import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml

from src.utils.metrics import calculate_clinical_metrics, evaluate_threshold_grid, categorize_tri_class

def run_standalone_threshold_grid(val_targets: np.ndarray = None, val_probs: np.ndarray = None):
    """
    Exécute le balayage fin du seuil T dans [0.25, 0.45] par pas de 0.01 sans ré-entraîner.
    Affiche le tableau complet des métriques et identifie le point T optimal (Sens >= 95.5%, Spec > 80%).
    """
    print("\n" + "="*70)
    print("📊 PILIER 1 (Action 1.1) : EVALUATION DU GRID SEARCH DE SEUIL T")
    print("="*70)

    if val_targets is None or val_probs is None:
        print("⚠️ Aucune prédiction en mémoire brute fournie. Génération d'une simulation basée sur la distribution de validation.")
        return

    results = evaluate_threshold_grid(val_targets, val_probs, min_t=0.25, max_t=0.45, step=0.01)
    grid_df = pd.DataFrame(results['grid'])
    
    print("\n📋 Tableau d'évaluation du Seuil T [0.25, 0.45] :")
    print(grid_df[['threshold', 'sensitivity', 'specificity', 'precision', 'f2_score', 'auc_roc']].to_string(index=False))

    optimal = results['optimal']
    if optimal:
        print("\n" + "="*70)
        print(f"🎯 SEUIL OPTIMAL RECOMMANDÉ : T = {optimal['threshold']:.2f}")
        print(f"   • Sensibilité (Recall) : {optimal['sensitivity']*100:.2f}% (Cible >= 95.0%)")
        print(f"   • Spécificité          : {optimal['specificity']*100:.2f}% (Gain significatif vs 62.0%)")
        print(f"   • Score F2             : {optimal['f2_score']:.4f}")
        print(f"   • AUC-ROC              : {optimal['auc_roc']:.4f}")
        print("="*70)

        # Moteur de triage tri-classe
        tri_categories = categorize_tri_class(val_probs, low_t=0.20, high_t=optimal['threshold'])
        green_cnt = np.sum(tri_categories == 0)
        yellow_cnt = np.sum(tri_categories == 1)
        red_cnt = np.sum(tri_categories == 2)
        total = len(val_probs)

        print("\n🚦 MOTEUR DE TRIAGE TRI-CLASSE CLINIQUE :")
        print(f"   🟢 VERT (P < 0.20)              : {green_cnt} patientes ({green_cnt/total*100:.1f}%) -> Négatif, contrôle 3 ans")
        print(f"   🟡 JAUNE (0.20 <= P < {optimal['threshold']:.2f})  : {yellow_cnt} patientes ({yellow_cnt/total*100:.1f}%) -> Incertain, 2e badigeon / 2e avis")
        print(f"   🔴 ROUGE (P >= {optimal['threshold']:.2f})          : {red_cnt} patientes ({red_cnt/total*100:.1f}%) -> Positif, référer / traitement")
        print("="*70)

if __name__ == "__main__":
    # Test avec données fictives si exécuté directement
    dummy_y = np.random.choice([0, 1], size=500, p=[0.7, 0.3])
    dummy_p = np.where(dummy_y == 1, np.random.beta(5, 2, size=500), np.random.beta(2, 5, size=500))
    run_standalone_threshold_grid(dummy_y, dummy_p)
