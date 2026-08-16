"""
Module de vérification de l'intégrité des images dans les jeux de données CSV (train, val, test).
Détecte et nettoie les fichiers JPEG corrompus ou tronqués ('Premature end of JPEG file').
"""

import os
import argparse
import pandas as pd
from PIL import Image, ImageFile
import cv2

# Permet de lever des exceptions lors de la détection d'images tronquées
ImageFile.LOAD_TRUNCATED_IMAGES = False

def check_image_integrity(filepath: str) -> bool:
    """
    Vérifie si une image est lisible et non corrompue via PIL et OpenCV.
    Retourne True si l'image est valide, False sinon.
    """
    if not os.path.exists(filepath):
        return False
    
    # 1. Test de lecture PIL
    try:
        with Image.open(filepath) as img:
            img.verify()
    except Exception:
        return False
    
    # Re-ouverture pour charger les données de pixels
    try:
        with Image.open(filepath) as img:
            img.load()
    except Exception:
        return False

    # 2. Test OpenCV
    try:
        mat = cv2.imread(filepath)
        if mat is None or mat.size == 0:
            return False
    except Exception:
        return False

    return True

def sanitize_csv(csv_path: str, output_csv_path: str = None) -> int:
    """
    Vérifie chaque ligne d'un CSV et supprime les fichiers corrompus.
    Retourne le nombre de fichiers corrompus supprimés.
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ Fichier CSV non trouvé : {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    if 'filepath' not in df.columns:
        print(f"❌ La colonne 'filepath' est absente de {csv_path}")
        return 0

    initial_count = len(df)
    valid_mask = []
    corrupted_files = []

    for idx, row in df.iterrows():
        fpath = str(row['filepath'])
        if check_image_integrity(fpath):
            valid_mask.append(True)
        else:
            valid_mask.append(False)
            corrupted_files.append(fpath)

    sanitized_df = df[valid_mask].copy()
    removed_count = initial_count - len(sanitized_df)

    save_path = output_csv_path if output_csv_path else csv_path
    sanitized_df.to_csv(save_path, index=False)

    print(f"📊 [{os.path.basename(csv_path)}] Total: {initial_count} | Valides: {len(sanitized_df)} | Purges: {removed_count}")
    if corrupted_files:
        print(f"   ⚠️ Fichiers corrompus détectés ({len(corrupted_files)}) :")
        for cf in corrupted_files[:5]:
            print(f"     - {cf}")
        if len(corrupted_files) > 5:
            print(f"     - ... et {len(corrupted_files) - 5} autres.")

    return removed_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitizer de dataset IVA JPEG")
    parser.add_argument("--data_dir", type=str, default="./data/processed", help="Dossier contenant train.csv, val.csv, test.csv")
    args = parser.parse_args()

    total_purged = 0
    for split in ["train.csv", "val.csv", "test.csv"]:
        csv_file = os.path.join(args.data_dir, split)
        if os.path.exists(csv_file):
            total_purged += sanitize_csv(csv_file)
        else:
            print(f"ℹ️ Aucun fichier {split} trouvé dans {args.data_dir}")

    print(f"✅ Vérification terminée. Total d'images corrompues supprimées : {total_purged}")
