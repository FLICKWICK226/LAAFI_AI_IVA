import os
import sys

def download_intel_mobileodt_dataset(target_data_path: str = "./data/raw") -> None:
    """
    Télécharge et extrait le dataset Kaggle 'intel-mobileodt-cervical-cancer-screening'
    de manière idempotente dans le répertoire cible.
    """
    os.makedirs(target_data_path, exist_ok=True)
    
    # Vérification de l'existence des données
    train_dir = os.path.join(target_data_path, "train")
    train_alt_dir = os.path.join(target_data_path, "train", "train")
    
    if (os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0) or \
       (os.path.exists(train_alt_dir) and len(os.listdir(train_alt_dir)) > 0):
        print(f"✅ Dataset déjà présent dans : {target_data_path} (Téléchargement ignoré)")
        return

    print("📥 Initialisation du téléchargement depuis Kaggle...")
    
    # Support transparent Colab userdata ou variables d'environnement
    if "google.colab" in sys.modules:
        try:
            from google.colab import userdata
            os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
            os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
            print("🔑 Identifiants Kaggle récupérés depuis google.colab.userdata")
        except Exception as e:
            print(f"⚠️ Impossible de lire les secrets Colab: {e}")

    assert "KAGGLE_USERNAME" in os.environ and os.environ["KAGGLE_USERNAME"], \
        "❌ Erreur: Variable d'environnement KAGGLE_USERNAME manquante."
    assert "KAGGLE_KEY" in os.environ and os.environ["KAGGLE_KEY"], \
        "❌ Erreur: Variable d'environnement KAGGLE_KEY manquante."

    import kaggle
    dataset_name = "intel-mobileodt-cervical-cancer-screening"
    
    print(f"📦 Téléchargement de {dataset_name} vers {target_data_path}...")
    kaggle.api.dataset_download_files(dataset_name, path=target_data_path, unzip=True)
    print(f"🎉 Téléchargement et extraction réussis dans : {target_data_path}")

if __name__ == "__main__":
    download_intel_mobileodt_dataset()
