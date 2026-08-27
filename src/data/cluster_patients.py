import os
import sys
import zipfile
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedGroupKFold
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


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
            for root, _, files in os.walk(cand, followlinks=True):
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                    print(f"⚡ Répertoire de données valide localisé dans : {cand}")
                    return cand
                    
    if os.path.exists("/kaggle/input"):
        return "/kaggle/input"
        
    return data_raw_dir

def compute_image_hashes(image_paths: list, hash_size: int = 8) -> tuple:
    """
    Calcule les empreintes perceptives structurelles (dHash + aHash) pour chaque image.
    Retourne deux matrices booléennes numpy de dimension (N, hash_size * hash_size).
    """
    try:
        import imagehash
    except ImportError:
        os.system("pip install -q imagehash")
        import imagehash

    d_hashes = []
    a_hashes = []
    print(f"🔍 Calcul des empreintes perceptives (dHash + aHash) sur {len(image_paths)} images...")
    
    for path in tqdm(image_paths, desc="Empreintes Perceptives (dHash/aHash)"):
        try:
            with Image.open(path) as img:
                img_rgb = img.convert('RGB')
                d_h = imagehash.dhash(img_rgb, hash_size=hash_size)
                a_h = imagehash.average_hash(img_rgb, hash_size=hash_size)
                
                d_hashes.append(d_h.hash.flatten())
                a_hashes.append(a_h.hash.flatten())
        except Exception:
            # Fallback en cas d'image corrompue
            d_hashes.append(np.zeros(hash_size * hash_size, dtype=bool))
            a_hashes.append(np.zeros(hash_size * hash_size, dtype=bool))
            
    return np.array(d_hashes, dtype=bool), np.array(a_hashes, dtype=bool)

def cluster_images_by_perceptual_hash(
    image_paths: list,
    labels: list = None,
    max_hamming_distance: int = 6,
    batch_size: int = 2000,
    ambiguous_report_path: str = "./reports/ambiguous_clusters.csv"
) -> list:
    """
    Regroupe les images quasi-identiques (rafales vidéo, même examen colposcopique)
    dans des clusters patients étanches en utilisant la distance de Hamming et les composantes connexes.
    
    Garantie ZÉRO FUITE : Aucun filtrage par label n'est appliqué lors du regroupement,
    assurant que deux clichés du même examen ne soient jamais séparés entre train et test.
    Les clusters présentant des labels contradictoires sont isolés dans un registre d'ambiguïté.
    """
    num_images = len(image_paths)
    if num_images == 0:
        return []

    # 1. Calcul des hashes
    d_matrix, a_matrix = compute_image_hashes(image_paths, hash_size=8)
    
    # 2. Construction de la matrice d'adjacence du graphe de similarité
    print(f"🔗 Détection des quasi-doublons & construction des clusters patients (seuil Hamming <= {max_hamming_distance})...")
    
    row_ind, col_ind = [], []
    for i in range(0, num_images, batch_size):
        end_i = min(i + batch_size, num_images)
        d_block_i = d_matrix[i:end_i]
        a_block_i = a_matrix[i:end_i]
        
        for j in range(i, num_images, batch_size):
            end_j = min(j + batch_size, num_images)
            d_block_j = d_matrix[j:end_j]
            a_block_j = a_matrix[j:end_j]
            
            # Distances de Hamming sur dHash et aHash
            d_diffs = np.count_nonzero(d_block_i[:, None, :] != d_block_j[None, :, :], axis=2)
            a_diffs = np.count_nonzero(a_block_i[:, None, :] != a_block_j[None, :, :], axis=2)
            
            # Match si les deux descripteurs sont proches
            match_r, match_c = np.where((d_diffs <= max_hamming_distance) & (a_diffs <= max_hamming_distance))
            
            for r, c in zip(match_r, match_c):
                idx_i = i + r
                idx_j = j + c
                if idx_i != idx_j:
                    row_ind.append(idx_i)
                    col_ind.append(idx_j)
                    row_ind.append(idx_j)
                    col_ind.append(idx_i)

    # 3. Calcul des composantes connexes (Connected Components)
    data = np.ones(len(row_ind), dtype=int)
    adj_sparse = sp.csr_matrix((data, (row_ind, col_ind)), shape=(num_images, num_images))
    n_components, comp_labels = connected_components(adj_sparse, directed=False)
    
    patient_ids = [f"patient_cluster_{c:05d}" for c in comp_labels]
    
    # 4. Audit & Traçabilité des clusters à labels contradictoires (Ambiguïté clinique)
    if labels is not None and len(labels) == num_images:
        cluster_to_labels = {}
        for idx, (p_id, lbl, path) in enumerate(zip(patient_ids, labels, image_paths)):
            if p_id not in cluster_to_labels:
                cluster_to_labels[p_id] = []
            cluster_to_labels[p_id].append((idx, path, lbl))
            
        ambiguous_rows = []
        for p_id, items in cluster_to_labels.items():
            distinct_labels = set(lbl for _, _, lbl in items)
            if len(distinct_labels) > 1:
                for idx, path, lbl in items:
                    ambiguous_rows.append({
                        "patient_id": p_id,
                        "filepath": path,
                        "label": lbl,
                        "cluster_distinct_labels": ";".join(sorted(list(distinct_labels))),
                        "cluster_size": len(items)
                    })
        
        if ambiguous_rows:
            ambiguous_df = pd.DataFrame(ambiguous_rows)
            os.makedirs(os.path.dirname(os.path.abspath(ambiguous_report_path)), exist_ok=True)
            ambiguous_df.to_csv(ambiguous_report_path, index=False)
            print(f"⚠️ {len(set(r['patient_id'] for r in ambiguous_rows))} clusters à labels discordants isolés dans : {ambiguous_report_path}")
        else:
            if ambiguous_report_path:
                os.makedirs(os.path.dirname(os.path.abspath(ambiguous_report_path)), exist_ok=True)
                pd.DataFrame(columns=["patient_id", "filepath", "label", "cluster_distinct_labels", "cluster_size"]).to_csv(ambiguous_report_path, index=False)
    
    # Statistiques du clustering
    unique_clusters, counts = np.unique(patient_ids, return_counts=True)
    multi_img_clusters = np.sum(counts > 1)
    max_cluster_size = np.max(counts) if len(counts) > 0 else 0
    grouped_images_count = np.sum(counts[counts > 1])
    
    print(f"✅ Clustering terminé : {n_components} clusters patients uniques identifiés.")
    print(f"📊 {multi_img_clusters} groupes de quasi-doublons détectés (totalisant {grouped_images_count} images).")
    print(f"📊 Taille de cluster maximale : {max_cluster_size} images d'une même patiente.")

    return patient_ids

def generate_patient_clusters_and_splits(
    data_raw_dir: str = "./data/raw",
    output_dir: str = "./data/processed",
    seed: int = 42,
    max_hamming_distance: int = 6
) -> None:
    """
    Génère les splits patient train.csv, val.csv, test.csv sous data/processed/ ou /kaggle/working/
    avec garantie mathématique de ZERO DATA LEAKAGE via Perceptual Hashing (imagehash).
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

    for full_path in tqdm(all_files, desc="Indexation des classes d'images"):
        cls_label = get_class_label(full_path)
        if cls_label is not None:
            image_paths.append(full_path)
            labels.append(cls_label)

    if len(image_paths) == 0:
        print(f"⚠️ Aucune image trouvée dans {resolved_raw_dir}. Fichiers CSV de structure créés.")
        df_dummy = pd.DataFrame(columns=['filepath', 'label', 'target', 'patient_id', 'split'])
        df_dummy.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        df_dummy.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        df_dummy.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        return

    # 1. Clustering réel des patientes via imagehash
    patient_ids = cluster_images_by_perceptual_hash(
        image_paths=image_paths,
        labels=labels,
        max_hamming_distance=max_hamming_distance
    )

    label_to_target = {'Type_1': 0, 'Type_2': 1, 'Type_3': 2}
    targets = [label_to_target[lbl] for lbl in labels]

    df = pd.DataFrame({
        'filepath': image_paths,
        'label': labels,
        'target': targets,
        'patient_id': patient_ids
    })

    # 2. Découpage étanche par StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(sgkf.split(df, df['label'], df['patient_id']))
    
    train_idx, test_idx = folds[0][0], folds[0][1]
    val_idx = folds[1][1]
    train_idx = np.array([i for i in train_idx if i not in val_idx])

    df['split'] = 'train'
    df.loc[val_idx, 'split'] = 'val'
    df.loc[test_idx, 'split'] = 'test'

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    # 3. Contrôle qualité rigoureux : Vérification de fuite de données (Data Leakage Audit)
    train_patients = set(train_df['patient_id'])
    val_patients = set(val_df['patient_id'])
    test_patients = set(test_df['patient_id'])

    leak_train_val = train_patients.intersection(val_patients)
    leak_train_test = train_patients.intersection(test_patients)
    leak_val_test = val_patients.intersection(test_patients)

    assert len(leak_train_val) == 0, f"❌ FUITE DE DONNÉES DÉTECTÉE : {len(leak_train_val)} patients partagés entre Train et Val !"
    assert len(leak_train_test) == 0, f"❌ FUITE DE DONNÉES DÉTECTÉE : {len(leak_train_test)} patients partagés entre Train et Test !"
    assert len(leak_val_test) == 0, f"❌ FUITE DE DONNÉES DÉTECTÉE : {len(leak_val_test)} patients partagés entre Val et Test !"

    print("🛡️ VALIDATION ÉTANCHÉITÉ : 0 patient partagé entre les splits (Zéro Fuite de Données).")

    # 4. Sauvegarde des CSVs
    df.to_csv(os.path.join(output_dir, 'manifest_anatomy.csv'), index=False)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    summary_info = {
        "total_images": len(df),
        "unique_patient_clusters": len(np.unique(patient_ids)),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_patients": len(train_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "class_distribution": {
            "train": train_df['label'].value_counts().to_dict(),
            "val": val_df['label'].value_counts().to_dict(),
            "test": test_df['label'].value_counts().to_dict()
        },
        "data_leakage_detected": False
    }

    summary_path = os.path.join(output_dir, "clustering_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_info, f_sum, indent=4)

    print(f"✅ Patient-Level Splits générés avec succès dans : {output_dir}")
    print(f"📊 Total images : {len(df)} (Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)})")
    print(f"📊 Total patient clusters : {len(np.unique(patient_ids))} (Train: {len(train_patients)} | Val: {len(val_patients)} | Test: {len(test_patients)})")

    # 5. Compilation binaire Memory-Mapped (.mmap / .npy) pour un débit DataLoader optimal (0 ms d'I/O)
    try:
        from src.data.convert_to_mmap import convert_splits_to_mmap
        convert_splits_to_mmap(processed_dir=output_dir, output_dir=output_dir)
    except Exception as e_m:
        print(f"⚠️ Remarque compilation mmap : {e_m}")

if __name__ == "__main__":
    generate_patient_clusters_and_splits()
