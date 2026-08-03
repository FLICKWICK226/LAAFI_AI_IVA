import os
import sys

def download_intel_mobileodt_dataset(target_data_path: str = "./data/raw") -> str:
    """
    Télécharge et extrait le dataset Kaggle 'intel-mobileodt-cervical-cancer-screening'
    ou détecte automatiquement son montage natif sous Kaggle (/kaggle/input/).
    """
    # 1. Détection automatique du montage natif Kaggle (/kaggle/input/)
    kaggle_inputs = [
        "/kaggle/input/intel-mobileodt-cervical-cancer-screening",
        "/kaggle/input/intel-mobileodt-cervical-cancer-screening/train/train"
    ]
    for k_in in kaggle_inputs:
        train_dir = os.path.join(k_in, "train")
        if os.path.exists(k_in) and os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
            print(f"⚡ Zero-Download Mode : Dataset monté nativement dans : {k_in}")
            return k_in

    os.makedirs(target_data_path, exist_ok=True)
    
    # Vérification de l'existence locale des données
    train_dir = os.path.join(target_data_path, "train")
    train_alt_dir = os.path.join(target_data_path, "train", "train")
    
    if (os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0) or \
       (os.path.exists(train_alt_dir) and len(os.listdir(train_alt_dir)) > 0):
        print(f"✅ Dataset déjà présent dans : {target_data_path} (Téléchargement ignoré)")
        return target_data_path

    print("📥 Initialisation de l'accès aux données via la clé API Colab/Kaggle...")

    if "google.colab" in sys.modules:
        try:
            from google.colab import userdata
            colab_api_key = None
            for key_name in ["KAGGLE_API_KEY", "KAGGLE_KEY", "KAGGLE_TOKEN"]:
                try:
                    colab_api_key = userdata.get(key_name)
                    if colab_api_key:
                        break
                except Exception:
                    continue

            if colab_api_key:
                os.environ["KAGGLE_API_KEY"] = colab_api_key
                os.environ["KAGGLE_KEY"] = colab_api_key
                print("🔑 Clé API Kaggle configurée depuis Colab Userdata !")
            else:
                print("⚠️ Secret 'KAGGLE_API_KEY' non trouvé dans Colab Userdata.")

            try:
                username = userdata.get("KAGGLE_USERNAME")
                if username:
                    os.environ["KAGGLE_USERNAME"] = username
            except Exception:
                pass

        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture du secret Colab : {e}")

    competition_name = "intel-mobileodt-cervical-cancer-screening"
    
    import kaggle
    import zipfile
    from tqdm import tqdm
    print(f"📦 Téléchargement des données vers {target_data_path}...")
    
    def extract_with_progress(zip_path, extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            print(f"📂 Extraction de {len(members)} fichiers depuis {os.path.basename(zip_path)}...")
            for member in tqdm(members, desc="Extraction Zip"):
                zip_ref.extract(member, extract_to)

    try:
        print(f"🔄 Téléchargement via l'API Kaggle officielle : {competition_name}...")
        kaggle.api.competition_download_files(competition_name, path=target_data_path)
        
        for item in os.listdir(target_data_path):
            if item.endswith(".zip"):
                zip_ref_path = os.path.join(target_data_path, item)
                extract_with_progress(zip_ref_path, target_data_path)
                os.remove(zip_ref_path)
                
        print(f"🎉 Téléchargement et extraction réussis dans : {target_data_path}")
    except Exception as e:
        print(f"❌ Échec de l'ingestion Kaggle API : {e}")
        print("💡 SI VOUS AVEZ UNE ERREUR 403 : Rendez-vous sur https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/rules et cliquez sur 'I Understand and Accept'.")

    return target_data_path

if __name__ == "__main__":
    download_intel_mobileodt_dataset()
