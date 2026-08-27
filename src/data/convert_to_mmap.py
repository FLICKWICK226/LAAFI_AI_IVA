"""
LAAFI_AI IVA - Memory-Mapped Dataset Converter (Phase C Optimization)
Compiles pre-split datasets (train, val, test) into high-performance binary memory-mapped arrays (.mmap/.npy).
Eliminates individual file I/O and OS filesystem lookups during model training.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def convert_single_split_to_mmap(
    csv_path: str,
    output_dir: str,
    split_name: str,
    img_size: tuple = (224, 224)
) -> dict:
    """
    Convertit un fichier CSV de split (train/val/test) en conteneur binaire NumPy memmap.
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ Fichier de split introuvable : {csv_path}. Conversion ignorée.")
        return {}

    df = pd.read_csv(csv_path)
    label_map = {'Type_1': 0, 'Type_2': 1, 'Type_3': 2}

    if 'label' in df.columns and len(df) > 0:
        valid_mask = df['label'].astype(str).isin(label_map.keys())
        df = df[valid_mask].copy()
        df['target'] = df['label'].map(label_map).astype(int)
    elif 'target' not in df.columns:
        df['target'] = 0

    num_samples = len(df)
    h, w = img_size

    os.makedirs(output_dir, exist_ok=True)
    mmap_img_path = os.path.join(output_dir, f"{split_name}_images.mmap")
    labels_path = os.path.join(output_dir, f"{split_name}_labels.npy")
    meta_path = os.path.join(output_dir, f"{split_name}_mmap_meta.json")

    print(f"\n📦 Conversion de {split_name} ({num_samples} images) vers {mmap_img_path}...")

    # Création du fichier binaire memmap sur disque
    mmap_images = np.memmap(
        mmap_img_path,
        dtype='uint8',
        mode='w+',
        shape=(num_samples, h, w, 3)
    )

    targets = np.zeros(num_samples, dtype=np.int64)
    patient_ids = []
    corrupted_count = 0

    for i in tqdm(range(num_samples), desc=f"Mmap {split_name}", unit="img"):
        row = df.iloc[i]
        img_path = str(row['filepath'])
        target = int(row['target'])
        patient_id = str(row.get('patient_id', f"unknown_{i}"))

        targets[i] = target
        patient_ids.append(patient_id)

        img = None
        if os.path.exists(img_path):
            img = cv2.imread(img_path)

        if img is not None and img.size > 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            mmap_images[i] = img
        else:
            corrupted_count += 1
            mmap_images[i] = np.zeros((h, w, 3), dtype=np.uint8)

    # Forcer l'écriture synchrone sur disque
    mmap_images.flush()
    del mmap_images

    # Sauvegarde des labels et métadonnées associées
    np.save(labels_path, targets)

    meta = {
        "split_name": split_name,
        "num_samples": num_samples,
        "img_shape": [h, w, 3],
        "dtype": "uint8",
        "corrupted_count": corrupted_count,
        "class_distribution": {int(k): int(v) for k, v in pd.Series(targets).value_counts().to_dict().items()},
        "patient_ids": patient_ids
    }

    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=4)

    file_size_mb = os.path.getsize(mmap_img_path) / (1024 * 1024)
    print(f"✅ {split_name} converti avec succès ({file_size_mb:.2f} Mo, {corrupted_count} fallback(s)).")
    return meta

def convert_splits_to_mmap(
    processed_dir: str = "./data/processed",
    output_dir: str = None,
    img_size: tuple = (224, 224)
) -> dict:
    """
    Exécute la compilation Memory-Mapped pour train.csv, val.csv et test.csv.
    """
    if output_dir is None:
        output_dir = processed_dir

    print("=" * 70)
    print("🚀 LAAFI_AI - Compilation des Datasets Memory-Mapped (.mmap / .npy)")
    print(f"📁 Répertoire source  : {processed_dir}")
    print(f"💾 Répertoire sortie  : {output_dir}")
    print(f"📐 Résolution standard: {img_size[0]}x{img_size[1]}")
    print("=" * 70)

    results = {}
    for split in ["train", "val", "test"]:
        csv_p = os.path.join(processed_dir, f"{split}.csv")
        if os.path.exists(csv_p):
            results[split] = convert_single_split_to_mmap(
                csv_path=csv_p,
                output_dir=output_dir,
                split_name=split,
                img_size=img_size
            )

    print("\n🎉 Compilation Memory-Mapped terminée pour tous les splits disponibles !")
    return results

if __name__ == "__main__":
    src_dir = sys.argv[1] if len(sys.argv) > 1 else "./data/processed"
    dst_dir = sys.argv[2] if len(sys.argv) > 2 else src_dir
    convert_splits_to_mmap(src_dir, dst_dir)
