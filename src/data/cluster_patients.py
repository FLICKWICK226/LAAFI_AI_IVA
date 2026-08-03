import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

def generate_patient_clusters_and_splits(
    data_raw_dir: str = "./data/raw",
    output_dir: str = "./data/processed",
    seed: int = 42
) -> None:
    """
    Génère les splits patient train.csv, val.csv, test.csv sous data/processed/ ou /kaggle/working/.
    """
    # Auto-détection de l'environnement Kaggle / Colab
    if os.path.exists("/kaggle/input/intel-mobileodt-cervical-cancer-screening"):
        data_raw_dir = "/kaggle/input/intel-mobileodt-cervical-cancer-screening"
        output_dir = "/kaggle/working/data/processed"
    elif os.path.exists("/content/data_fast"):
        output_dir = "./data/processed"

    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = []
    labels = []
    
    print(f"🔍 Indexation et recherche des images depuis : {data_raw_dir}...")
    all_files = []
    for root, _, files in os.walk(data_raw_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))

    for full_path in tqdm(all_files, desc="Indexation des images"):
        parent_dir = os.path.basename(os.path.dirname(full_path))
        if parent_dir in ['Type_1', 'Type_2', 'Type_3']:
            image_paths.append(full_path)
            labels.append(parent_dir)

    if len(image_paths) == 0:
        print(f"⚠️ Aucune image trouvée dans {data_raw_dir}. Création de fichiers CSV de structure.")
        df_dummy = pd.DataFrame(columns=['filepath', 'label', 'patient_id', 'split'])
        df_dummy.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        df_dummy.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        df_dummy.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        return

    df = pd.DataFrame({'filepath': image_paths, 'label': labels})
    
    def extract_patient_id(path):
        filename = os.path.basename(path)
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        if len(parts) >= 2 and (parts[0].isdigit() or parts[0].isalnum()):
            return f"patient_{parts[0]}"
        return f"patient_{base_name}"

    df['patient_id'] = df['filepath'].apply(extract_patient_id)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(sgkf.split(df, df['label'], df['patient_id']))
    
    train_idx, test_idx = folds[0][0], folds[0][1]
    val_idx = folds[1][1]
    train_idx = np.array([i for i in train_idx if i not in val_idx])

    df['split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"✅ Patient-Level Splits générés dans : {output_dir}")
    print(f"📊 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

if __name__ == "__main__":
    generate_patient_clusters_and_splits()
