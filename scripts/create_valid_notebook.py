import json
import os

def build_master_kaggle_notebook():
    notebook_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🔬 LAAFI_AI IVA Engine - Master Pipeline Kaggle Native\n",
                    "**Projet :** Dépistage du Cancer du Col de l'Utérus par Imagerie Smartphone (IVA/VIA)\n",
                    "**Spécifications :** SaMD Class II CADe/CADx | Directives OMS / IFCPC | Zero-Download Mode\n",
                    "\n",
                    "---"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📥 Cellule 1 : Initialisation & Synchronisation du Dépôt\n",
                    "Clonage ou mise à jour automatique (`git pull`) du code source et installation des dépendances."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, sys\n",
                    "\n",
                    "repo_path = \"/kaggle/working/LAAFI_AI_IVA\"\n",
                    "if not os.path.exists(repo_path):\n",
                    "    print(\"📥 Clonage initial du dépôt GitHub officiel...\")\n",
                    "    !git clone https://github.com/FLICKWICK226/LAAFI_AI_IVA.git {repo_path}\n",
                    "else:\n",
                    "    print(\"🔄 Dépôt déjà présent : mise à jour avec les derniers commits GitHub (git pull)...\")\n",
                    "    !git -C {repo_path} pull origin main\n",
                    "\n",
                    "%cd {repo_path}\n",
                    "if repo_path not in sys.path:\n",
                    "    sys.path.append(repo_path)\n",
                    "\n",
                    "# Installation des dépendances\n",
                    "!pip install -q timm albumentations opencv-python-headless matplotlib pandas scikit-learn tqdm py7zr onnx imagehash"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🖥️ Cellule 2 : Détection Hardware GPU & Mode Zero-Download\n",
                    "Vérification de l'accélérateur GPU (Nvidia T4 x2 recommandé) et détection du jeu de données pré-monté sous `/kaggle/input/`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "\n",
                    "print(f\"PyTorch Version : {torch.__version__}\")\n",
                    "if torch.cuda.is_available():\n",
                    "    gpu_name = torch.cuda.get_device_name(0)\n",
                    "    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9\n",
                    "    print(f\"🚀 GPU Kaggle Détecté : {gpu_name} ({vram_gb:.2f} GB VRAM)\")\n",
                    "else:\n",
                    "    print(\"⚠️ GPU non détecté. Assurez-vous d'activer l'accélérateur GPU dans le menu 'Settings' -> 'Accelerator' de Kaggle.\")\n",
                    "\n",
                    "input_path = \"/kaggle/input/competitions/intel-mobileodt-cervical-cancer-screening\"\n",
                    "if not os.path.exists(input_path):\n",
                    "    input_path = \"/kaggle/input/intel-mobileodt-cervical-cancer-screening\"\n",
                    "\n",
                    "if os.path.exists(input_path):\n",
                    "    print(f\"⚡ Zero-Download Mode Actif ! Jeu de données détecté sous : {input_path}\")\n",
                    "else:\n",
                    "    print(f\"⚠️ Dataset non trouvé. Pensez à cliquer sur '+ Add Data' dans le panneau de droite sur Kaggle.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🎨 Cellule 3 : Génération Hors-ligne des Masques Perlin\n",
                    "Pré-génération des masques de bruit procédural (sang et glaire) dans le dossier réscriptible `/kaggle/working/data/synthetic_masks`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import importlib\n",
                    "import src.data.generate_perlin_masks\n",
                    "importlib.reload(src.data.generate_perlin_masks)\n",
                    "from src.data.generate_perlin_masks import generate_perlin_masks\n",
                    "\n",
                    "generate_perlin_masks(\n",
                    "    output_dir=\"/kaggle/working/data/synthetic_masks\",\n",
                    "    num_masks=1000\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🔍 Cellule 4 : Indexation Native & Splits Patients Étanches (Zéro Fuite)\n",
                    "Indexation universelle multi-chemins des images d'entrée depuis `/kaggle/input/` et découpage sans fuite de données via dHash/aHash (Hamming <= 6)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import importlib\n",
                    "import src.data.cluster_patients\n",
                    "importlib.reload(src.data.cluster_patients)\n",
                    "from src.data.cluster_patients import generate_patient_clusters_and_splits\n",
                    "\n",
                    "generate_patient_clusters_and_splits(\n",
                    "    data_raw_dir=\"/kaggle/input/competitions/intel-mobileodt-cervical-cancer-screening\",\n",
                    "    output_dir=\"/kaggle/working/data/processed\"\n",
                    ")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 🏋️ Cellule 5 : Entraînement Optimisé Stage 2 (ConvNeXt-Small 3-Classes)\n",
                    "Lancement du moteur d'entraînement unifié avec Précision Mixte (AMP), Warmup Backbone Freeze, Early Stopping et Gradient Accumulation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import importlib\n",
                    "import src.train\n",
                    "importlib.reload(src.train)\n",
                    "from src.train import train_laafi_ai_model\n",
                    "\n",
                    "train_laafi_ai_model(config_path=\"./config/config.yaml\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📊 Cellule 6 : Évaluation Réelle & Exportation des Métriques Biomédicales\n",
                    "Évaluation du meilleur checkpoint sur le Test Set : Matrice de Confusion IFCPC 3-Classes, Courbe ROC Triage OMS et rapports CSV/JSON."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import json\n",
                    "import torch\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from torch.utils.data import DataLoader\n",
                    "from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay\n",
                    "\n",
                    "from src.data.dataset import IVADataset\n",
                    "from src.models.classifier_lesion import IVALesionClassifierStage2\n",
                    "from src.utils.metrics import calculate_anatomical_metrics, calculate_clinical_triage_metrics\n",
                    "\n",
                    "fig_dir = \"/kaggle/working/outputs/figures\"\n",
                    "rep_dir = \"/kaggle/working/outputs/reports\"\n",
                    "os.makedirs(fig_dir, exist_ok=True)\n",
                    "os.makedirs(rep_dir, exist_ok=True)\n",
                    "\n",
                    "history_path = os.path.join(rep_dir, \"training_history.csv\")\n",
                    "if os.path.exists(history_path):\n",
                    "    df_hist = pd.read_csv(history_path)\n",
                    "    print(\"📊 Historique d'entraînement résumé :\")\n",
                    "    display(df_hist.tail(10))\n",
                    "    \n",
                    "    fig, ax1 = plt.subplots(figsize=(8, 4))\n",
                    "    ax1.set_xlabel('Epoch')\n",
                    "    ax1.set_ylabel('Train Loss', color='tab:red')\n",
                    "    ax1.plot(df_hist['epoch'], df_hist['train_loss'], color='tab:red', marker='o', label='Train Loss')\n",
                    "    \n",
                    "    ax2 = ax1.twinx()\n",
                    "    ax2.set_ylabel('Val AUC & F2', color='tab:blue')\n",
                    "    ax2.plot(df_hist['epoch'], df_hist['val_auc'], color='tab:blue', marker='s', label='Val AUC')\n",
                    "    ax2.plot(df_hist['epoch'], df_hist['val_f2_score'], color='tab:green', marker='^', label='Val F2')\n",
                    "    \n",
                    "    plt.title(\"LAAFI_AI Kaggle Engine - Courbes d'Entraînement\")\n",
                    "    fig.tight_layout()\n",
                    "    plt.savefig(os.path.join(fig_dir, \"learning_curves.png\"), dpi=150)\n",
                    "    plt.show()\n",
                    "    plt.close()\n",
                    "\n",
                    "# Évaluation aveugle sur le Test Set\n",
                    "test_csv = \"/kaggle/working/data/processed/test.csv\"\n",
                    "if not os.path.exists(test_csv):\n",
                    "    test_csv = \"./data/processed/test.csv\"\n",
                    "\n",
                    "ckpt_path = \"/kaggle/working/models/checkpoints/best_model.pt\"\n",
                    "if not os.path.exists(ckpt_path):\n",
                    "    ckpt_path = \"./models/checkpoints/best_model.pt\"\n",
                    "\n",
                    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                    "if os.path.exists(test_csv) and os.path.exists(ckpt_path):\n",
                    "    test_ds = IVADataset(csv_file=test_csv, is_train=False)\n",
                    "    if len(test_ds) > 0:\n",
                    "        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)\n",
                    "        model = IVALesionClassifierStage2(backbone_name=\"convnext_small\", pretrained=False, num_classes=3).to(device)\n",
                    "        ckpt = torch.load(ckpt_path, map_location=device)\n",
                    "        state_dict = ckpt.get('model_state_dict', ckpt)\n",
                    "        clean_state = {k.replace(\"_orig_mod.\", \"\"): v for k, v in state_dict.items()}\n",
                    "        model.load_state_dict(clean_state, strict=False)\n",
                    "        model.eval()\n",
                    "        \n",
                    "        all_targets, all_probs = [], []\n",
                    "        with torch.no_grad():\n",
                    "            for imgs, targets, _ in test_loader:\n",
                    "                imgs = imgs.to(device)\n",
                    "                logits = model(imgs)\n",
                    "                probs = torch.softmax(logits, dim=1).cpu().numpy()\n",
                    "                all_probs.append(probs)\n",
                    "                all_targets.append(targets.numpy())\n",
                    "                \n",
                    "        all_probs = np.vstack(all_probs)\n",
                    "        all_targets = np.concatenate(all_targets)\n",
                    "        \n",
                    "        # 1. Métriques anatomiques IFCPC\n",
                    "        anat_metrics = calculate_anatomical_metrics(all_targets, all_probs)\n",
                    "        print(\"📋 Métriques Anatomiques (Type 1 / Type 2 / Type 3) :\")\n",
                    "        print(f\"   • Accuracy globale : {anat_metrics['accuracy']*100:.2f}%\")\n",
                    "        print(f\"   • Macro F1-Score   : {anat_metrics['macro_f1']:.4f}\")\n",
                    "        print(f\"   • Macro AUC-ROC    : {anat_metrics['macro_auc_roc']:.4f}\")\n",
                    "        \n",
                    "        cm = np.array(anat_metrics['confusion_matrix'])\n",
                    "        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Type 1', 'Type 2', 'Type 3'])\n",
                    "        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))\n",
                    "        disp.plot(cmap=plt.cm.Blues, ax=ax_cm)\n",
                    "        plt.title(\"Matrice de Confusion Anatomique 3-Classes (IFCPC)\")\n",
                    "        plt.savefig(os.path.join(fig_dir, \"confusion_matrix.png\"), dpi=150)\n",
                    "        plt.show()\n",
                    "        plt.close()\n",
                    "        \n",
                    "        # 2. Triage Clinique SaMD (Directives OMS)\n",
                    "        triage_metrics = calculate_clinical_triage_metrics(all_targets, all_probs, referral_threshold=0.35)\n",
                    "        print(\"\\n🚦 Moteur de Triage Clinique SaMD (Directives OMS) :\")\n",
                    "        print(f\"   • Triage Accuracy          : {triage_metrics['triage_accuracy']*100:.2f}%\")\n",
                    "        print(f\"   • Sensibilité Éligibles    : {triage_metrics['sensitivity_eligible']*100:.2f}%\")\n",
                    "        print(f\"   • Spécificité Sécurité T3  : {triage_metrics['safety_specificity_type3']*100:.2f}%\")\n",
                    "        print(f\"   • Triage AUC-ROC           : {triage_metrics['triage_auc_roc']:.4f}\")\n",
                    "        \n",
                    "        with open(os.path.join(rep_dir, \"clinical_triage_report.json\"), \"w\", encoding=\"utf-8\") as f_tr:\n",
                    "            json.dump(triage_metrics, f_tr, indent=4)\n",
                    "            \n",
                    "        # Courbe ROC Triage\n",
                    "        y_true_eligible = (all_targets != 2).astype(int)\n",
                    "        prob_eligible = all_probs[:, 0] + all_probs[:, 1]\n",
                    "        if len(np.unique(y_true_eligible)) > 1:\n",
                    "            fpr, tpr, _ = roc_curve(y_true_eligible, prob_eligible)\n",
                    "            roc_auc_val = auc(fpr, tpr)\n",
                    "            plt.figure(figsize=(6, 5))\n",
                    "            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Triage (AUC = {roc_auc_val:.4f})')\n",
                    "            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
                    "            plt.xlabel('1 - Spécificité')\n",
                    "            plt.ylabel('Sensibilité')\n",
                    "            plt.title('Courbe ROC - Triage Éligibilité Traitement Local')\n",
                    "            plt.legend(loc=\"lower right\")\n",
                    "            plt.savefig(os.path.join(fig_dir, \"roc_curve.png\"), dpi=150)\n",
                    "            plt.show()\n",
                    "            plt.close()\n",
                    "            \n",
                    "        # Sauvegarde du rapport CSV de métriques réelles\n",
                    "        metrics_df = pd.DataFrame([{\n",
                    "            \"metric_name\": \"Accuracy Globale\", \"value\": f\"{anat_metrics['accuracy']*100:.2f}%\"\n",
                    "        }, {\n",
                    "            \"metric_name\": \"Macro F1-Score\", \"value\": f\"{anat_metrics['macro_f1']:.4f}\"\n",
                    "        }, {\n",
                    "            \"metric_name\": \"Macro AUC-ROC\", \"value\": f\"{anat_metrics['macro_auc_roc']:.4f}\"\n",
                    "        }, {\n",
                    "            \"metric_name\": \"Triage Accuracy (OMS)\", \"value\": f\"{triage_metrics['triage_accuracy']*100:.2f}%\"\n",
                    "        }, {\n",
                    "            \"metric_name\": \"Sécurité Type 3 Référé\", \"value\": f\"{triage_metrics['safety_specificity_type3']*100:.2f}%\"\n",
                    "        }])\n",
                    "        metrics_df.to_csv(os.path.join(rep_dir, \"metrics_report.csv\"), index=False)\n",
                    "        print(f\"📄 Rapport de métriques sauvegardé dans : {os.path.join(rep_dir, 'metrics_report.csv')}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 👁️ Cellule 7 : Audit Visuel Explicable Grad-CAM\n",
                    "Validation des cartes d'attention visuelle pour s'assurer que le réseau identifie la Zone de Transformation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import torch\n",
                    "import importlib\n",
                    "import src.utils.visualization\n",
                    "importlib.reload(src.utils.visualization)\n",
                    "from src.utils.visualization import generate_gradcam_heatmap\n",
                    "from src.data.dataset import IVADataset\n",
                    "from src.models.classifier_lesion import IVALesionClassifierStage2\n",
                    "\n",
                    "val_csv = \"/kaggle/working/data/processed/val.csv\"\n",
                    "if not os.path.exists(val_csv):\n",
                    "    val_csv = \"./data/processed/val.csv\"\n",
                    "\n",
                    "if os.path.exists(val_csv):\n",
                    "    val_ds = IVADataset(csv_file=val_csv, is_train=False)\n",
                    "    if len(val_ds) > 0:\n",
                    "        sample_tensor, sample_target, _ = val_ds[0]\n",
                    "        device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                    "        model = IVALesionClassifierStage2(backbone_name=\"convnext_small\", pretrained=False, num_classes=3).to(device)\n",
                    "        ckpt_path = \"/kaggle/working/models/checkpoints/best_model.pt\"\n",
                    "        if not os.path.exists(ckpt_path):\n",
                    "            ckpt_path = \"./models/checkpoints/best_model.pt\"\n",
                    "        if os.path.exists(ckpt_path):\n",
                    "            ckpt = torch.load(ckpt_path, map_location=device)\n",
                    "            state_dict = ckpt.get('model_state_dict', ckpt)\n",
                    "            new_state_dict = {k.replace(\"_orig_mod.\", \"\"): v for k, v in state_dict.items()}\n",
                    "            model.load_state_dict(new_state_dict, strict=False)\n",
                    "            print(f\"💾 Poids entraînés {ckpt_path} chargés avec succès.\")\n",
                    "        output_fig = \"/kaggle/working/outputs/figures/gradcam_sample.png\"\n",
                    "        generate_gradcam_heatmap(\n",
                    "            model=model,\n",
                    "            image_tensor=sample_tensor,\n",
                    "            output_path=output_fig\n",
                    "        )\n",
                    "        print(f\"✅ Carte d'attention Grad-CAM générée sous : {output_fig}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 📦 Cellule 8 : Exportation ONNX Final\n",
                    "Exportation du meilleur modèle sous `./models/exported/best_model.onnx` pour inférence mobile de terrain."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os\n",
                    "import importlib\n",
                    "import src.models.export_onnx\n",
                    "importlib.reload(src.models.export_onnx)\n",
                    "from src.models.export_onnx import export_model_to_onnx\n",
                    "\n",
                    "ckpt_path = \"/kaggle/working/models/checkpoints/best_model.pt\"\n",
                    "if not os.path.exists(ckpt_path):\n",
                    "    ckpt_path = \"./models/checkpoints/best_model.pt\"\n",
                    "\n",
                    "output_onnx = \"/kaggle/working/models/exported/best_model.onnx\"\n",
                    "if not os.path.exists(os.path.dirname(output_onnx)):\n",
                    "    output_onnx = \"./models/exported/best_model.onnx\"\n",
                    "\n",
                    "export_model_to_onnx(\n",
                    "    checkpoint_path=ckpt_path,\n",
                    "    output_onnx_path=output_onnx,\n",
                    "    img_size=(224, 224),\n",
                    "    backbone_name=\"convnext_small\"\n",
                    ")\n",
                    "print(\"🎉 PIPELINE KAGGLE EXÉCUTÉ ET MODÈLE ONNX EXPORTÉ AVEC SUCCÈS !\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    target_paths = [
        "./notebooks/LAAFI_AI_IVA_Kaggle_Master_Pipeline.ipynb",
        "./notebooks/laafi-ai-via.ipynb"
    ]

    for path in target_paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook_dict, f, indent=1, ensure_ascii=False)
        print(f"Notebook JSON valide genere : {path}")

if __name__ == "__main__":
    build_master_kaggle_notebook()
