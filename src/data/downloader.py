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

    # Configuration exclusive depuis Colab userdata
    if "google.colab" in sys.modules:
        try:
            from google.colab import userdata
            colab_api_key = userdata.get("KAGGLE_API_KEY") or userdata.get("KAGGLE_TOKEN")
            if colab_api_key:
                os.environ["KAGGLE_API_KEY"] = colab_api_key
                os.environ["KAGGLE_KEY"] = colab_api_key
                username = userdata.get("KAGGLE_USERNAME")
                if username:
                    os.environ["KAGGLE_USERNAME"] = username
                print("🔑 Clé API Kaggle configurée avec succès depuis Colab Userdata (KAGGLE_API_KEY) !")
            else:
                print("⚠️ Secret 'KAGGLE_API_KEY' non trouvé dans Colab Userdata.")
        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture du secret Colab : {e}")

    dataset_name = "intel-mobileodt-cervical-cancer-screening"
    
    import kaggle
    print(f"📦 Téléchargement de {dataset_name} vers {target_data_path}...")
    kaggle.api.dataset_download_files(dataset_name, path=target_data_path, unzip=True)
    print(f"🎉 Téléchargement et extraction réussis dans : {target_data_path}")

if __name__ == "__main__":
    download_intel_mobileodt_dataset()
