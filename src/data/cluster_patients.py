import os
import zipfile
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold

def extract_archives_if_needed(target_dir: str) -> None:
    """
    Extrait automatiquement les archives .zip ou .7z si aucune image non compressée n'est présente.
    """
    try:
        import py7zr
    except ImportError:
        os.system("pip install -q py7zr")
        import py7zr

    for root, _, files in os.walk(target_dir, followlinks=True):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith('.zip'):
                print(f"📦 Extraction de l'archive ZIP : {file}...")
                with zipfile.ZipFile(full_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
            elif file.endswith('.7z'):
                print(f"📦 Extraction de l'archive 7Z : {file}...")
                try:
                    with py7zr.SevenZipFile(full_path, mode='r') as z:
                        z.extractall(path=target_dir)
                except Exception as e:
                    print(f"⚠️ Impossible d'extraire {file} : {e}")

def get_class_label(path: str) -> str:
    """
    Détermine la classe ('Type_1', 'Type_2', 'Type_3') à partir du chemin complet de l'image.
    """
    norm_path = path.replace("\\", "/")
    parts = norm_path.split("/")
    
    # 1. Vérification exacte du nom du dossier parent ou d'un segment
    for p in reversed(parts[:-1]):
        if p in ['Type_1', 'Type_2', 'Type_3']:
            return p
            
    # 2. Vérification insensible à la casse et sous-chaînes (ex: additional_Type_1_v2)
    for p in reversed(parts[:-1]):
        p_lower = p.lower()
        if 'type_1' in p_lower or 'type1' in p_lower:
            return 'Type_1'
        elif 'type_2' in p_lower or 'type2' in p_lower:
            return 'Type_2'
        elif 'type_3' in p_lower or 'type3' in p_lower:
            return 'Type_3'
            
    return None

def resolve_raw_data_dir(data_raw_dir: str) -> str:
    """
    Résoluteur universel multi-chemins pour localiser le jeu de données d'entrée.
    """
    candidates = [
        data_raw_dir,
        "/kaggle/input/competitions/intel-mobileodt-cervical-cancer-screening",
        "/kaggle/input/intel-mobileodt-cervical-cancer-screening",
        "/kaggle/input"
    ]
    for cand in candidates:
        if os.path.exists(cand):
            # Vérifier si des fichiers images existent sous ce répertoire
            for root, _, files in os.walk(cand, followlinks=True):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    print(f"⚡ Repertoire de donnees valide localise dans : {cand}")
                    return cand
                    
    if os.path.exists("/kaggle/input"):
        return "/kaggle/input"
        
    return data_raw_dir

def generate_patient_clusters_and_splits(
    data_raw_dir: str = "./data/raw",
    output_dir: str = "./data/processed",
    seed: int = 42
) -> None:
    """
    Génère les splits patient train.csv, val.csv, test.csv sous data/processed/ ou /kaggle/working/.
    """
    resolved_raw_dir = resolve_raw_data_dir(data_raw_dir)
    
    # Auto-détection du dossier de sortie sous Kaggle / Colab
    if os.path.exists("/kaggle/working"):
        output_dir = "/kaggle/working/data/processed"
    elif os.path.exists("/content/data_fast"):
        output_dir = "./data/processed"

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🔍 Indexation et recherche des images depuis : {resolved_raw_dir}...")
    
    image_paths = []
    labels = []
    all_files = []
    
    for root, _, files in os.walk(resolved_raw_dir, followlinks=True):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))

    # Si aucune image brute n'est trouvée, chercher et extraire les archives .zip / .7z
    if len(all_files) == 0 and os.path.exists("/kaggle/working"):
        print("⚠️ Aucune image décompressée trouvée. Tentative d'extraction des archives .zip/.7z...")
        extract_archives_if_needed(resolved_raw_dir)
        
        all_files = []
        for root, _, files in os.walk(resolved_raw_dir, followlinks=True):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(root, f))

    for full_path in tqdm(all_files, desc="Parsing des étiquettes d'images"):
        cls_label = get_class_label(full_path)
        if cls_label is not None:
            image_paths.append(full_path)
            labels.append(cls_label)

    if len(image_paths) == 0:
        print(f"⚠️ Aucune image trouvée dans {resolved_raw_dir}. Fichiers CSV de structure créés.")
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

    print(f"✅ Patient-Level Splits générés avec succès dans : {output_dir}")
    print(f"📊 Total images indexées : {len(df)} (Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)})")

if __name__ == "__main__":
    generate_patient_clusters_and_splits()
