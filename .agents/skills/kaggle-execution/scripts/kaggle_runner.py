#!/usr/bin/env python3
"""
Kaggle Execution & CI/CD ML Helper Script for Antigravity Agent.
Automates authentication checks, kernel metadata validation, remote execution,
real-time status polling, and artifact downloading.
"""

import os
import sys
import json
import time
import argparse
import base64
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "0"

if sys.platform == 'win32':
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
elif hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def get_kaggle_api():
    """Initialise et authentifie l'API Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        sys.stderr.write(f"❌ Erreur d'authentification Kaggle API : {e}\n")
        sys.stderr.write("Vérifiez que le fichier ~/.kaggle/kaggle.json existe avec vos identifiants valides.\n")
        sys.exit(1)

def cmd_check_auth(args):
    """Vérifie l'état d'authentification Kaggle."""
    api = get_kaggle_api()
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    username = "unknown"
    if os.path.exists(kaggle_json_path):
        try:
            with open(kaggle_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                username = data.get('username', 'unknown')
        except Exception:
            pass

    result = {
        "status": "authenticated",
        "username": username,
        "credentials_path": kaggle_json_path
    }
    
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f_out:
            json.dump(result, f_out, indent=2)
    print(f"✅ Authentification Kaggle active pour l'utilisateur : {username}")
    print(json.dumps(result, indent=2))

def cmd_list_kernels(args):
    """Liste les notebooks de l'utilisateur sur Kaggle."""
    api = get_kaggle_api()
    user = args.user
    if not user:
        kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(kaggle_json_path):
            try:
                with open(kaggle_json_path, 'r', encoding='utf-8') as f:
                    user = json.load(f).get('username')
            except Exception:
                pass

    kernels = api.kernels_list(user=user)
    
    kernel_list = []
    for k in kernels:
        ref = getattr(k, 'ref', str(k))
        title = getattr(k, 'title', '')
        last_run = getattr(k, 'last_run_time', getattr(k, 'lastRunTime', 'N/A'))
        kernel_list.append({
            "ref": ref,
            "title": title,
            "last_run_time": str(last_run)
        })

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f_out:
            json.dump(kernel_list, f_out, indent=2)

    print(f"📊 {len(kernel_list)} notebooks trouvés pour '{user or 'utilisateur courant'}' :")
    for item in kernel_list:
        print(f"  - [{item['title']}] ({item['ref']}) | Dernier run : {item['last_run_time']}")

def cmd_push(args):
    """Valide les métadonnées et pousse le notebook sur Kaggle."""
    api = get_kaggle_api()
    kernel_dir = os.path.abspath(args.kernel_dir)
    metadata_path = os.path.join(kernel_dir, "kernel-metadata.json")

    if not os.path.exists(metadata_path):
        sys.stderr.write(f"❌ Fichier de métadonnées introuvable : {metadata_path}\n")
        sys.exit(1)

    with open(metadata_path, 'r', encoding='utf-8') as f_meta:
        meta = json.load(f_meta)

    code_file = meta.get('code_file', '')
    code_path = os.path.join(kernel_dir, code_file)
    if not os.path.exists(code_path):
        sys.stderr.write(f"❌ Fichier de code spécifié introuvable : {code_path}\n")
        sys.exit(1)

    print(f"🚀 Poussée du kernel '{meta.get('id')}' depuis {kernel_dir}...")
    try:
        api.kernels_push(kernel_dir)
        print(f"✅ Kernel poussé avec succès sur Kaggle !")
        print(f"📍 URL : https://www.kaggle.com/code/{meta.get('id')}")
    except Exception as e:
        sys.stderr.write(f"❌ Erreur lors du push Kaggle : {e}\n")
        sys.exit(1)

def cmd_status(args):
    """Vérifie le statut d'exécution d'un kernel distant."""
    api = get_kaggle_api()
    kernel_id = args.kernel_id
    
    try:
        status_resp = api.kernels_status(kernel_id)
        status = status_resp.get('status', 'UNKNOWN') if isinstance(status_resp, dict) else str(status_resp)
        failure_msg = status_resp.get('failureMessage') if isinstance(status_resp, dict) else None
        
        res = {
            "kernel_id": kernel_id,
            "status": status,
            "failure_message": failure_msg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w', encoding='utf-8') as f_out:
                json.dump(res, f_out, indent=2)

        print(f"📡 Statut Kernel '{kernel_id}' : {status}")
        if failure_msg:
            print(f"⚠️ Message d'échec : {failure_msg}")
    except Exception as e:
        sys.stderr.write(f"❌ Erreur interrogation statut : {e}\n")
        sys.exit(1)

def cmd_pull_outputs(args):
    """Télécharge les fichiers de sortie produits par un run Kaggle."""
    api = get_kaggle_api()
    kernel_id = args.kernel_id
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📥 Téléchargement des outputs du kernel '{kernel_id}' vers {output_dir}...")
    try:
        api.kernels_output(kernel_id, path=output_dir, force=True)
        downloaded = []
        for root, _, files in os.walk(output_dir):
            for f in files:
                p = os.path.join(root, f)
                downloaded.append({"file": f, "path": p, "size_kb": os.path.getsize(p)/1024.0})
                print(f"  [+] {f} ({os.path.getsize(p)/1024.0:.1f} KB)")

        print(f"✅ {len(downloaded)} fichiers téléchargés avec succès dans {output_dir}.")
    except Exception as e:
        sys.stderr.write(f"❌ Erreur téléchargement outputs : {e}\n")
        sys.exit(1)

def cmd_extract_figures(args):
    """Extrait les figures PNG (base64) contenues dans un notebook .ipynb exécuté."""
    notebook_path = os.path.abspath(args.notebook_path)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(notebook_path):
        sys.stderr.write(f"❌ Notebook introuvable : {notebook_path}\n")
        sys.exit(1)

    with open(notebook_path, 'r', encoding='utf-8') as f_nb:
        nb = json.load(f_nb)

    count = 0
    for i, cell in enumerate(nb.get('cells', [])):
        for j, out in enumerate(cell.get('outputs', [])):
            data = out.get('data', {})
            if 'image/png' in data:
                count += 1
                b64_data = data['image/png']
                img_path = os.path.join(output_dir, f"figure_cell_{i+1}_{count}.png")
                with open(img_path, 'wb') as f_img:
                    f_img.write(base64.b64decode(b64_data))
                print(f"  🖼️ Figure extraite : {img_path} ({os.path.getsize(img_path)/1024.0:.1f} KB)")

    print(f"✅ Total {count} figures extraites dans {output_dir}.")

def main():
    parser = argparse.ArgumentParser(description="Kaggle CI/CD ML CLI Runner for Antigravity")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check-auth
    p_auth = subparsers.add_parser("check-auth", help="Vérifie l'authentification Kaggle")
    p_auth.add_argument("--output", help="Chemin du fichier JSON de sortie")

    # list-kernels
    p_list = subparsers.add_parser("list-kernels", help="Liste les notebooks de l'utilisateur")
    p_list.add_argument("--user", default=None, help="Nom d'utilisateur Kaggle (optionnel)")
    p_list.add_argument("--output", help="Chemin du fichier JSON de sortie")

    # push
    p_push = subparsers.add_parser("push", help="Pousse un notebook pour exécution distante sur GPU")
    p_push.add_argument("--kernel-dir", default="./notebooks", help="Dossier contenant le notebook et kernel-metadata.json")

    # status
    p_stat = subparsers.add_parser("status", help="Interroge le statut d'un kernel distant")
    p_stat.add_argument("--kernel-id", required=True, help="Identifiant complet (ex: username/slug)")
    p_stat.add_argument("--output", help="Chemin du fichier JSON de sortie")

    # pull-outputs
    p_pull = subparsers.add_parser("pull-outputs", help="Télécharge les artefacts d'un kernel exécuté")
    p_pull.add_argument("--kernel-id", required=True, help="Identifiant complet (ex: username/slug)")
    p_pull.add_argument("--output-dir", default="./outputs/kaggle_remote", help="Dossier de destination")

    # extract-figures
    p_fig = subparsers.add_parser("extract-figures", help="Extrait les figures PNG d'un notebook exécuté")
    p_fig.add_argument("--notebook-path", required=True, help="Chemin du fichier .ipynb exécuté")
    p_fig.add_argument("--output-dir", default="./outputs/figures", help="Dossier de destination des images")

    args = parser.parse_args()

    if args.command == "check-auth":
        cmd_check_auth(args)
    elif args.command == "list-kernels":
        cmd_list_kernels(args)
    elif args.command == "push":
        cmd_push(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "pull-outputs":
        cmd_pull_outputs(args)
    elif args.command == "extract-figures":
        cmd_extract_figures(args)

if __name__ == "__main__":
    main()
