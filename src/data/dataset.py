import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data.augmentations import FastPerlinNoiseLoader, build_iva_augmentation_pipeline

class IVADataset(Dataset):
    """
    Dataset PyTorch hautement optimisé pour l'imagerie IVA du col de l'utérus.
    Résout automatiquement les chemins Google Drive FUSE et Kaggle vers le SSD local rapide.
    """
    def __init__(
        self,
        csv_file: str,
        is_train: bool = True,
        masks_dir: str = "./data/synthetic_masks",
        perlin_proba: float = 0.30
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

        self.perlin_loader = FastPerlinNoiseLoader(masks_dir=masks_dir)
        self.transform = build_iva_augmentation_pipeline(is_train=is_train)
        self.cache = {} # Cache RAM pour les images 224x224 (0 ms d'I/O après le 1er accès)
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
        # 1. Branche Memory-Mapped Ultra-Rapide (0 appel système, lecture directe mémoire OS)
        if self.mmap_images is not None and self.targets is not None:
            base_image = np.array(self.mmap_images[idx], copy=True)
            target = int(self.targets[idx])
            patient_id = self.patient_ids[idx] if self.patient_ids and idx < len(self.patient_ids) else 'unknown'
        else:
            row = self.df.iloc[idx]
            img_path = str(row['filepath'])
            target = int(row['target'])
            patient_id = row.get('patient_id', 'unknown')

            # Récupération directe depuis le Cache RAM si présent (0 ms I/O)
            if img_path in self.cache:
                base_image = self.cache[img_path]
            else:
                image = None
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                
                if image is not None and image.size > 0:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if image.shape[0] != 224 or image.shape[1] != 224:
                        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                    base_image = image
                else:
                    base_image = np.zeros((224, 224, 3), dtype=np.uint8)
                
                self.cache[img_path] = base_image

            base_image = base_image.copy()

        # 2. Injection dynamique de masques de bruit biologiques (uniquement en train)
        if self.is_train and np.random.rand() < self.perlin_proba:
            noise_type = np.random.choice(['blood', 'mucus'])
            base_image = self.perlin_loader.add_blood_or_mucus(base_image, noise_type=noise_type)

        # 3. Transformation Albumentations ultra-rapide
        augmented = self.transform(image=base_image)
        image_tensor = torch.as_tensor(augmented['image']).permute(2, 0, 1).float()

        return image_tensor, torch.tensor(target, dtype=torch.long), patient_id

if __name__ == "__main__":
    dataset = IVADataset(csv_file="./data/processed/train.csv", is_train=True)
    print(f"📊 IVADataset initialisé avec {len(dataset)} échantillons.")
