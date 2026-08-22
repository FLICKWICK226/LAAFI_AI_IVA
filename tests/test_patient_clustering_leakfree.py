import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from PIL import Image

from src.data.cluster_patients import (
    compute_image_hashes,
    cluster_images_by_perceptual_hash,
    generate_patient_clusters_and_splits,
    get_class_label
)
from src.utils.metrics import evaluate_threshold_grid, calculate_clinical_metrics

def test_get_class_label():
    assert get_class_label("/kaggle/input/intel/train/Type_1/10.jpg") == "Type_1"
    assert get_class_label("/kaggle/input/intel/additional_Type_2_v2/100.jpg") == "Type_2"
    assert get_class_label("C:\\data\\raw\\Type_3\\test.png") == "Type_3"
    assert get_class_label("/data/unknown/image.jpg") is None

def test_perceptual_hash_clustering():
    temp_dir = tempfile.mkdtemp()
    try:
        # Création de 6 images de test :
        # - 2 images quasi-identiques Type_1 (patient A)
        # - 2 images quasi-identiques Type_2 (patient B)
        # - 2 images aléatoires distinctes Type_3 (patients C et D)
        paths = []
        labels = []

        # Image A1
        img_a1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img_a1[20:80, 20:80] = 255
        p_a1 = os.path.join(temp_dir, "a1.jpg")
        Image.fromarray(img_a1).save(p_a1)
        paths.append(p_a1); labels.append("Type_1")

        # Image A2 (légère variation de luminosité)
        img_a2 = np.clip(img_a1.astype(int) + 5, 0, 255).astype(np.uint8)
        p_a2 = os.path.join(temp_dir, "a2.jpg")
        Image.fromarray(img_a2).save(p_a2)
        paths.append(p_a2); labels.append("Type_1")

        # Image B1
        img_b1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img_b1[:, 50:] = 200
        p_b1 = os.path.join(temp_dir, "b1.jpg")
        Image.fromarray(img_b1).save(p_b1)
        paths.append(p_b1); labels.append("Type_2")

        # Image B2
        img_b2 = np.clip(img_b1.astype(int) + 2, 0, 255).astype(np.uint8)
        p_b2 = os.path.join(temp_dir, "b2.jpg")
        Image.fromarray(img_b2).save(p_b2)
        paths.append(p_b2); labels.append("Type_2")

        # Image C (bruit aléatoire)
        np.random.seed(42)
        img_c = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        p_c = os.path.join(temp_dir, "c.jpg")
        Image.fromarray(img_c).save(p_c)
        paths.append(p_c); labels.append("Type_3")

        # Image D (dégradé)
        x = np.linspace(0, 255, 100, dtype=np.uint8)
        img_d = np.repeat(x[None, :, None], 100, axis=0)
        img_d = np.repeat(img_d, 3, axis=2)
        p_d = os.path.join(temp_dir, "d.jpg")
        Image.fromarray(img_d).save(p_d)
        paths.append(p_d); labels.append("Type_3")

        patient_ids = cluster_images_by_perceptual_hash(paths, labels, max_hamming_distance=6)

        # Vérifications
        assert patient_ids[0] == patient_ids[1], "Les images quasi-identiques A1 et A2 doivent être dans le même cluster patient !"
        assert patient_ids[2] == patient_ids[3], "Les images quasi-identiques B1 et B2 doivent être dans le même cluster patient !"
        assert patient_ids[0] != patient_ids[2], "Des images de classes différentes ne doivent pas être dans le même cluster !"

    finally:
        shutil.rmtree(temp_dir)

def test_leak_free_split_generation():
    temp_raw = tempfile.mkdtemp()
    temp_out = tempfile.mkdtemp()
    try:
        # Création d'une arborescence factice
        for cls in ['Type_1', 'Type_2', 'Type_3']:
            cls_dir = os.path.join(temp_raw, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(15):
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(img).save(os.path.join(cls_dir, f"img_{i}.jpg"))

        generate_patient_clusters_and_splits(
            data_raw_dir=temp_raw,
            output_dir=temp_out,
            seed=42
        )

        train_df = pd.read_csv(os.path.join(temp_out, "train.csv"))
        val_df = pd.read_csv(os.path.join(temp_out, "val.csv"))
        test_df = pd.read_csv(os.path.join(temp_out, "test.csv"))

        # Vérification de l'étanchéité stricte des splits
        train_p = set(train_df['patient_id'])
        val_p = set(val_df['patient_id'])
        test_p = set(test_df['patient_id'])

        assert len(train_p.intersection(val_p)) == 0, "Fuite de données entre Train et Val !"
        assert len(train_p.intersection(test_p)) == 0, "Fuite de données entre Train et Test !"
        assert len(val_p.intersection(test_p)) == 0, "Fuite de données entre Val et Test !"

    finally:
        shutil.rmtree(temp_raw)
        shutil.rmtree(temp_out)

def test_evaluate_threshold_grid():
    y_true = np.array([1]*50 + [0]*50)
    # Probabilités bien calibrées
    y_prob = np.array([0.8]*48 + [0.4]*2 + [0.1]*45 + [0.7]*5)

    res = evaluate_threshold_grid(
        y_true, y_prob,
        min_t=0.05, max_t=0.95, step=0.05,
        target_sensitivity=0.95, min_specificity=0.50
    )

    opt = res['optimal']
    assert opt is not None
    assert opt['sensitivity'] >= 0.95
    assert opt['specificity'] >= 0.50
    assert 0.05 <= opt['threshold'] <= 0.95
