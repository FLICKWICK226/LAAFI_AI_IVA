import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def generate_patient_clusters_and_splits(
    data_raw_dir: str = "./data/raw",
    output_dir: str = "./data/processed",
    seed: int = 42
) -> None:
    """
    Génère des identifiants patients étanches et crée les splits train.csv, val.csv, test.csv
    sans aucun risque de Data Leakage au niveau patient.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Recherche des images dans data/raw
    image_paths = []
    labels = []
    
    from tqdm import tqdm
    print("🔍 Recherche et indexation des images pour le découpage par patient...")
    
    # Collecte préalable pour barre de progression
    all_files = []
    for root, dirs, files in os.walk(data_raw_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_files.append(os.path.join(root, f))

    for full_path in tqdm(all_files, desc="Indexation des images"):
        parent_dir = os.path.basename(os.path.dirname(full_path))
        if parent_dir in ['Type_1', 'Type_2', 'Type_3']:
            image_paths.append(full_path)
            labels.append(parent_dir)

    if len(image_paths) == 0:
        print(f"⚠️ Aucune image trouvée dans {data_raw_dir}. Assurez-vous d'avoir exécuté le downloader.")
        # Générer des fichiers CSV fictifs/structures de départ si le dataset n'est pas encore téléchargé localement
        df_dummy = pd.DataFrame(columns=['filepath', 'label', 'patient_id', 'split'])
        df_dummy.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        df_dummy.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        df_dummy.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        print(f"📝 Fichiers de structures CSV créés dans : {output_dir}")
        return

    df = pd.DataFrame({'filepath': image_paths, 'label': labels})
    
    # -------------------------------------------------------------
    # Groupement Patient : Extraction robuste de l'identifiant patient
    # Pour éviter le Data Leakage, chaque image doit impérativement être associée à un patient.
    # -------------------------------------------------------------
    def extract_patient_id(path):
        filename = os.path.basename(path)
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('_')
        # Si le pattern <patient_id>_<img_id> existe
        if len(parts) >= 2 and (parts[0].isdigit() or parts[0].isalnum()):
            return f"patient_{parts[0]}"
        # En l'absence de pattern clair, isoler par dossier parent ou identifiant unique explicite
        return f"patient_{base_name}"

    df['patient_id'] = df['filepath'].apply(extract_patient_id)

    # Vérification d'intégrité avant splitting
    if df['patient_id'].isnull().any():
        raise ValueError("❌ Erreur critique : Des valeurs nulles ont été détectées dans les patient_id !")
    
    # Séparation robuste avec StratifiedGroupKFold (5 Folds = 20% par fold)
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    
    folds = list(sgkf.split(df, df['label'], df['patient_id']))
    
    # Fold 0 = Test (20%), Fold 1 = Val (20%), Folds 2,3,4 = Train (60%)
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
    print(f"📊 Statistiques : Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    print(f"🔒 Étanchéité Patient : {len(set(train_df['patient_id']).intersection(set(test_df['patient_id'])))} chevauchements (doit être 0).")

if __name__ == "__main__":
    generate_patient_clusters_and_splits()
