import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.data.augmentations import FastPerlinNoiseLoader, build_iva_augmentation_pipeline

class IVADataset(Dataset):
    """
    Dataset PyTorch personnalisable pour l'imagerie IVA du col de l'utérus.
    Intègre les augmentations optiques et l'injection dynamique de masques de bruit biologique.
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
        self.perlin_loader = FastPerlinNoiseLoader(masks_dir=masks_dir)
        self.transform = build_iva_augmentation_pipeline(is_train=is_train)
        
        # Chargement du dataframe CSV
        if os.path.exists(csv_file):
            self.df = pd.read_csv(csv_file)
        else:
            self.df = pd.DataFrame(columns=['filepath', 'label', 'patient_id'])

        # Mappage des labels anatomiques en entiers
        label_map = {'Type_1': 0, 'Type_2': 1, 'Type_3': 2}
        if 'label' in self.df.columns and len(self.df) > 0:
            self.df['target'] = self.df['label'].map(lambda x: label_map.get(str(x), 0))
        else:
            self.df['target'] = 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        target = row['target']
        patient_id = row.get('patient_id', 'unknown')

        # Lecture de l'image (BGR -> RGB)
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Image factice de remplacement si le dataset n'est pas encore téléchargé
            image = np.zeros((384, 384, 3), dtype=np.uint8)

        # Injection dynamique de masques de bruit biologiques (uniquement en train)
        if self.is_train and np.random.rand() < self.perlin_proba:
            noise_type = np.random.choice(['blood', 'mucus'])
            image = self.perlin_loader.add_blood_or_mucus(image, noise_type=noise_type)

        # Transformation Albumentations
        augmented = self.transform(image=image)
        image_tensor = torch.tensor(augmented['image']).permute(2, 0, 1).float()

        return image_tensor, torch.tensor(target, dtype=torch.long), patient_id

if __name__ == "__main__":
    dataset = IVADataset(csv_file="./data/processed/train.csv", is_train=True)
    print(f"📊 IVADataset initialisé avec {len(dataset)} échantillons.")
