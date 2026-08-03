import os
import sys

def download_intel_mobileodt_dataset(target_data_path: str = "./data/raw") -> None:
    """
    Télécharge et extrait le dataset Kaggle 'intel-mobileodt-cervical-cancer-screening'
    de manière flexible et idempotente (support API Key, Token, kaggle.json, kagglehub, etc.).
    """
    os.makedirs(target_data_path, exist_ok=True)
    
    # Vérification de l'existence des données
    train_dir = os.path.join(target_data_path, "train")
    train_alt_dir = os.path.join(target_data_path, "train", "train")
    
    if (os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0) or \
       (os.path.exists(train_alt_dir) and len(os.listdir(train_alt_dir)) > 0):
        print(f"✅ Dataset déjà présent dans : {target_data_path} (Téléchargement ignoré)")
        return

    print("📥 Initialisation de l'accès aux données Kaggle...")

    # 1. Colab Userdata Secret Handling (Support multi-clés : API Key, Token, Username/Key)
    if "google.colab" in sys.modules:
        try:
            from google.colab import userdata
            for key_name in ["KAGGLE_API_KEY", "KAGGLE_TOKEN", "KAGGLE_KEY", "KAGGLE_USERNAME"]:
                val = userdata.get(key_name)
                if val:
                    os.environ[key_name] = val
                    if key_name in ["KAGGLE_API_KEY", "KAGGLE_TOKEN"]:
                        os.environ["KAGGLE_KEY"] = val # Fallback Kaggle API
            print("🔑 Clés/Tokens d'API Kaggle configurés depuis Colab Userdata.")
        except Exception as e:
            print(f"ℹ️ Colab userdata non utilisé ou partiel : {e}")

    # 2. Vérification de la présence de kaggle.json local ou global
    kaggle_json_paths = [
        "./kaggle.json",
        os.path.expanduser("~/.kaggle/kaggle.json"),
        os.path.join(os.environ.get("KAGGLE_CONFIG_DIR", ""), "kaggle.json")
    ]
    has_kaggle_json = any(os.path.exists(p) for p in kaggle_json_paths if p)

    # 3. Vérification des variables d'environnement (API Key / Token OU Username + Key)
    has_api_key = bool(os.environ.get("KAGGLE_API_KEY") or os.environ.get("KAGGLE_TOKEN"))
    has_credentials = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

    if not (has_kaggle_json or has_api_key or has_credentials):
        print("⚠️ Aucune clé d'authentification Kaggle trouvée.")
        print("💡 Veuillez fournir au choix :")
        print("   - Une clé API : KAGGLE_API_KEY ou KAGGLE_TOKEN")
        print("   - Des identifiants : KAGGLE_USERNAME + KAGGLE_KEY")
        print("   - Un fichier : kaggle.json dans le répertoire courant ou ~/.kaggle/")

    # Syntaxe d'ingestion Kaggle
    dataset_name = "intel-mobileodt-cervical-cancer-screening"
    
    # Stratégie 1 : kagglehub si disponible
    try:
        import kagglehub
        print(f"📦 Téléchargement via kagglehub ({dataset_name})...")
        path = kagglehub.dataset_download(f"intel/{dataset_name}")
        print(f"🎉 Dataset téléchargé via kagglehub dans : {path}")
        return
    except Exception:
        pass

    # Stratégie 2 : kaggle API standard
    try:
        import kaggle
        print(f"📦 Téléchargement via Kaggle API ({dataset_name}) vers {target_data_path}...")
        kaggle.api.dataset_download_files(dataset_name, path=target_data_path, unzip=True)
        print(f"🎉 Téléchargement et extraction réussis dans : {target_data_path}")
    except Exception as e:
        print(f"❌ Échec de l'ingestion Kaggle API : {e}")

if __name__ == "__main__":
    download_intel_mobileodt_dataset()
