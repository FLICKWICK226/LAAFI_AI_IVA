import os
import sys

def download_intel_mobileodt_dataset(target_data_path: str = "./data/raw") -> None:
    """
    Télécharge et extrait le dataset Kaggle 'intel-mobileodt-cervical-cancer-screening'
    en utilisant exclusivement la clé API Colab 'KAGGLE_API_KEY'.
    """
    os.makedirs(target_data_path, exist_ok=True)
    
    # Vérification de l'existence des données
    train_dir = os.path.join(target_data_path, "train")
    train_alt_dir = os.path.join(target_data_path, "train", "train")
    
    if (os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0) or \
       (os.path.exists(train_alt_dir) and len(os.listdir(train_alt_dir)) > 0):
        print(f"✅ Dataset déjà présent dans : {target_data_path} (Téléchargement ignoré)")
        return

    print("📥 Initialisation de l'accès aux données via la clé API Colab...")

    # Configuration exclusive depuis Colab userdata (sans exception sur secret manquant)
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
    dataset_mirror = "FLICKWICK/intel-mobileodt-cervical-cancer-screening"
    
    import kaggle
    import zipfile
    from tqdm import tqdm
    print(f"📦 Téléchargement des données vers {target_data_path}...")
    
    def extract_with_progress(zip_path, extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            print(f"📂 Extractions de {len(members)} fichiers depuis {os.path.basename(zip_path)}...")
            for member in tqdm(members, desc="Extraction Zip"):
                zip_ref.extract(member, extract_to)

    # 1. Tentative via la compétition officielle Kaggle
    try:
        print(f"🔄 Tentative via la compétition officielle Kaggle : {competition_name}...")
        kaggle.api.competition_download_files(competition_name, path=target_data_path)
        
        # Dézippage avec barre de progression
        for item in os.listdir(target_data_path):
            if item.endswith(".zip"):
                zip_ref_path = os.path.join(target_data_path, item)
                extract_with_progress(zip_ref_path, target_data_path)
                os.remove(zip_ref_path)
                
        print(f"🎉 Téléchargement et extraction de la compétition réussis dans : {target_data_path}")
        return
    except Exception as e:
        print(f"⚠️ Compétition inaccessible ({e}), tentative via la commande shell Kaggle...")

    # 2. Tentative de secours via la commande CLI Kaggle officielle
    try:
        os.system(f"kaggle competitions download -c {competition_name} -p {target_data_path}")
        for item in os.listdir(target_data_path):
            if item.endswith(".zip"):
                zip_ref_path = os.path.join(target_data_path, item)
                extract_with_progress(zip_ref_path, target_data_path)
                os.remove(zip_ref_path)
        print(f"🎉 Téléchargement réussi via la commande CLI dans : {target_data_path}")
    except Exception as e_cli:
        print(f"❌ Erreur lors du téléchargement : {e_cli}")
        print("💡 SI VOUS AVEZ UNE ERREUR 403 : Rendez-vous sur https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/rules et cliquez sur 'I Understand and Accept'.")

if __name__ == "__main__":
    download_intel_mobileodt_dataset()
