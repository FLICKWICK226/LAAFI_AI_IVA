import os
import tempfile
import shutil
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from src.data.cluster_patients import cluster_images_by_perceptual_hash, generate_patient_clusters_and_splits
from src.utils.metrics import calculate_clinical_triage_metrics, calculate_anatomical_metrics

def test_stratified_group_kfold_zero_leakage_all_folds():
    """Garantie mathématique de ZÉRO fuite patient sur les 5 folds de StratifiedGroupKFold."""
    num_patients = 50
    images_per_patient = 4
    total_samples = num_patients * images_per_patient

    patient_ids = []
    labels = []
    filepaths = []
    
    # 50 patientes réparties sur les 3 classes
    for i in range(num_patients):
        p_id = f"pat_{i:03d}"
        lbl = f"Type_{(i % 3) + 1}"
        for j in range(images_per_patient):
            patient_ids.append(p_id)
            labels.append(lbl)
            filepaths.append(f"img_{p_id}_{j}.jpg")

    df = pd.DataFrame({
        "filepath": filepaths,
        "label": labels,
        "patient_id": patient_ids
    })

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(df, df["label"], groups=df["patient_id"])):
        train_p = set(df.iloc[train_idx]["patient_id"])
        test_p = set(df.iloc[test_idx]["patient_id"])

        # Aucune patiente ne doit être partagée entre train et test
        leakage = train_p.intersection(test_p)
        assert len(leakage) == 0, f"Fuite détectée sur le fold {fold_idx} : {leakage}"

        # Chaque fold doit contenir les 3 classes anatomiques
        assert set(df.iloc[train_idx]["label"].unique()) == {"Type_1", "Type_2", "Type_3"}
        assert set(df.iloc[test_idx]["label"].unique()) == {"Type_1", "Type_2", "Type_3"}

def test_perceptual_hash_groups_identical_images_with_conflicting_labels():
    """
    Vérifie que 2 images quasi-identiques (même patiente) portant des labels contradictoires
    (ex: Type_1 vs Type_2 par erreur d'annotation) SONT REGROUPÉES dans le même cluster patient
    (pour éviter la fuite de données) et consignées dans reports/ambiguous_clusters.csv.
    """
    temp_dir = tempfile.mkdtemp()
    ambiguous_csv = os.path.join(temp_dir, "ambiguous_clusters.csv")
    try:
        # 2 images quasi-identiques
        img_base = np.zeros((100, 100, 3), dtype=np.uint8)
        img_base[30:70, 30:70] = 200

        p1 = os.path.join(temp_dir, "img1.jpg")
        p2 = os.path.join(temp_dir, "img2.jpg")
        Image.fromarray(img_base).save(p1)
        Image.fromarray(np.clip(img_base.astype(int) + 2, 0, 255).astype(np.uint8)).save(p2)

        # Labels contradictoires sur le même cliché
        paths = [p1, p2]
        labels = ["Type_1", "Type_2"]

        patient_ids = cluster_images_by_perceptual_hash(
            image_paths=paths,
            labels=labels,
            max_hamming_distance=6,
            ambiguous_report_path=ambiguous_csv
        )

        # 1. Zéro fuite : Doivent avoir le même patient_id
        assert patient_ids[0] == patient_ids[1], "Les quasi-doublons avec labels discordants DOIVENT être groupés ensemble !"

        # 2. Traçabilité : L'ambiguïté doit être inscrite dans le rapport CSV
        assert os.path.exists(ambiguous_csv), "Le fichier reports/ambiguous_clusters.csv doit être généré."
        df_amb = pd.read_csv(ambiguous_csv)
        assert len(df_amb) == 2
        assert df_amb["cluster_distinct_labels"].iloc[0] == "Type_1;Type_2"
    finally:
        shutil.rmtree(temp_dir)

def test_who_clinical_triage_decision_rules():
    """
    Vérifie la logique de décision du triage clinique OMS / IFCPC :
    - Type 1 et Type 2 sont éligibles au traitement ablatif local (True Label = 1).
    - Type 3 est inéligible au traitement local (True Label = 0) et doit être référé.
    - Si P(Type 3) >= seuil (0.35), la décision oriente vers la référence chirurgicale (y_pred_eligible = 0).
    """
    # 3 patientes : Type 1 (0), Type 2 (1), Type 3 (2)
    y_true = np.array([0, 1, 2])
    
    # Prédictions du modèle
    y_probs = np.array([
        [0.80, 0.15, 0.05],  # Patiente 1 : Type 1 clair -> Éligible
        [0.10, 0.75, 0.15],  # Patiente 2 : Type 2 clair -> Éligible
        [0.05, 0.20, 0.75],  # Patiente 3 : Type 3 clair -> Référé
    ])

    metrics = calculate_clinical_triage_metrics(y_true, y_probs, referral_threshold=0.35)
    
    assert metrics["triage_accuracy"] == 1.0
    assert metrics["sensitivity_eligible"] == 1.0
    assert metrics["safety_specificity_type3"] == 1.0
    assert metrics["confusion_matrix_2x2"]["true_referred_type3"] == 1
    assert metrics["confusion_matrix_2x2"]["true_eligible"] == 2
    assert metrics["confusion_matrix_2x2"]["false_eligible_risk"] == 0
