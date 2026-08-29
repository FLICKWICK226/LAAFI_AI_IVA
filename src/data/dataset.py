import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional
from src.preprocessing.cervix_pipeline import CervicalImagePipeline

class IVADataset(Dataset):
    """
    Dataset PyTorch pour l'imagerie IVA du col de l'utérus.
    Délègue l'ensemble des transformations et de l'ingestion à CervicalImagePipeline.
    Prend en charge les conteneurs Memory-Mapped binaires (.mmap/.npy) pour une vitesse I/O maximale.
    """
    def __init__(
        self,
        csv_file: str,
        is_train: bool = True,
        masks_dir: str = "./data/synthetic_masks",
        perlin_proba: float = 0.30,
        pipeline: Optional[CervicalImagePipeline] = None
    ):
        self.is_train = is_train
        self.perlin_proba = perlin_proba
        
        # Redirection automatique vers SSD local Colab / Kaggle si présent
        if os.path.exists("/kaggle/working/data/processed"):
            csv_file_kaggle = os.path.join("/kaggle/working/data/processed", os.path.basename(csv_file))
            if os.path.exists(csv_file_kaggle):
                csv_file = csv_file_kaggle

        if os.path.exists("/kaggle/working/data/synthetic_masks"):
            masks_dir = "/kaggle/working/data/synthetic_masks"

        if os.path.exists("/content/data_fast/processed"):
            csv_file_fast = csv_file.replace("./data", "/content/data_fast").replace("/content/drive/MyDrive/LAAFI_AI_IVA/data", "/content/data_fast")
            if os.path.exists(csv_file_fast):
                csv_file = csv_file_fast

        if os.path.exists("/content/data_fast/synthetic_masks"):
            masks_dir = "/content/data_fast/synthetic_masks"

        # Injection de dépendance du pipeline de prétraitement deep
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = CervicalImagePipeline(
                img_size=(224, 224),
                perlin_proba=perlin_proba,
                masks_dir=masks_dir
            )

        self.mmap_images = None
        self.targets = None
        self.patient_ids = None

        # Détection automatique d'un conteneur binaire Memory-Mapped (.mmap / .npy)
        split_dir = os.path.dirname(csv_file) if os.path.exists(csv_file) else "./data/processed"
        split_base = os.path.basename(csv_file).replace(".csv", "")
        mmap_img_path = os.path.join(split_dir, f"{split_base}_images.mmap")
        mmap_lbl_path = os.path.join(split_dir, f"{split_base}_labels.npy")
        mmap_meta_path = os.path.join(split_dir, f"{split_base}_mmap_meta.json")

        if os.path.exists(mmap_img_path) and os.path.exists(mmap_lbl_path):
            try:
                self.targets = np.load(mmap_lbl_path)
                num_samples = len(self.targets)
                self.mmap_images = np.memmap(
                    mmap_img_path,
                    dtype='uint8',
                    mode='r',
                    shape=(num_samples, 224, 224, 3)
                )
                if os.path.exists(mmap_meta_path):
                    with open(mmap_meta_path, 'r', encoding='utf-8') as f_m:
                        self.patient_ids = json.load(f_m).get('patient_ids', [])
                print(f"⚡ Mode Memory-Mapped Actif ({split_base}) : {num_samples} images mappées à 0 ms d'I/O.")
            except Exception as e_mmap:
                print(f"⚠️ Erreur chargement mmap ({e_mmap}). Fallback sur CSV.")
                self.mmap_images = None

        # Chargement du dataframe CSV (si pas de mmap ou fallback)
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
        else:
            self.df = pd.DataFrame(columns=['filepath', 'label', 'patient_id'])

        # Mappage des labels anatomiques en entiers
        label_map = {'Type_1': 0, 'Type_2': 1, 'Type_3': 2}
        if 'label' in self.df.columns and len(self.df) > 0:
            valid_mask = self.df['label'].astype(str).isin(label_map.keys())
            if not valid_mask.all():
                self.df = self.df[valid_mask].copy()
            self.df['target'] = self.df['label'].map(label_map).astype(int)
        elif 'target' not in self.df.columns:
            self.df['target'] = 0

        # Remplacement dynamique des chemins d'accès vers le SSD local /content/data_fast si présent
        if os.path.exists("/content/data_fast"):
            self.df['filepath'] = self.df['filepath'].apply(
                lambda p: str(p).replace("/content/drive/MyDrive/LAAFI_AI_IVA/data", "/content/data_fast")
                                .replace("./data", "/content/data_fast")
            )

    def __len__(self) -> int:
        if self.mmap_images is not None and self.targets is not None:
            return len(self.targets)
        return len(self.df)

    def __getitem__(self, idx: int):
        # 1. Résolution de la source de l'image (mmap ou chemin de fichier)
        if self.mmap_images is not None and self.targets is not None:
            raw_input = np.array(self.mmap_images[idx], copy=True)
            target = int(self.targets[idx])
            patient_id = self.patient_ids[idx] if self.patient_ids and idx < len(self.patient_ids) else 'unknown'
        else:
            row = self.df.iloc[idx]
            raw_input = str(row['filepath'])
            target = int(row['target'])
            patient_id = row.get('patient_id', 'unknown')

        # 2. Délégation intégrale au deep module CervicalImagePipeline
        image_tensor = self.pipeline.process(raw_input, is_train=self.is_train)

        return image_tensor, torch.tensor(target, dtype=torch.long), patient_id

if __name__ == "__main__":
    dataset = IVADataset(csv_file="./data/processed/train.csv", is_train=True)
    print(f"📊 IVADataset initialisé avec {len(dataset)} échantillons.")
