import pytest
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def test_stratified_group_kfold_no_leakage(dummy_patient_dataframe):
    """
    Vérifie avec rigueur mathématique qu'aucune patiente (patient_id)
    n'apparaît simultanément dans le train set et le validation/test set.
    """
    df = dummy_patient_dataframe
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    
    for train_idx, val_idx in sgkf.split(df, df["label"], groups=df["patient_id"]):
        train_patients = set(df.iloc[train_idx]["patient_id"].unique())
        val_patients = set(df.iloc[val_idx]["patient_id"].unique())
        
        # L'intersection des ensembles de patientes DOIT être strictement vide
        intersection = train_patients.intersection(val_patients)
        assert len(intersection) == 0, f"Fuite de données détectée pour les patientes : {intersection}"
