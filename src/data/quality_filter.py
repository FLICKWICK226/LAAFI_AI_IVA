import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from typing import Dict, Tuple, List

# Empêcher le crash sur les images tronquées tout en capturant l'anomalie
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CervicalImageQualityFilter:
    """
    Module de Contrôle Qualité Technique & Quarantaine pour images cervicales IVA (Non-clinique).
    Évalue :
    1. L'intégrité du fichier (détection des corruptions JPEG).
    2. Le niveau de netteté (variance du Laplacien).
    3. La saturation de l'exposition (flash LED brûlé).
    4. La sous-exposition (image trop sombre).
    """
    def __init__(
        self,
        min_laplacian_var: float = 40.0,
        max_overexposed_ratio: float = 0.25,
        max_underexposed_ratio: float = 0.35,
        min_resolution: Tuple[int, int] = (200, 200)
    ):
        self.min_laplacian_var = min_laplacian_var
        self.max_overexposed_ratio = max_overexposed_ratio
        self.max_underexposed_ratio = max_underexposed_ratio
        self.min_resolution = min_resolution

    def evaluate_image(self, image_path: str) -> Dict[str, any]:
        """
        Évalue une image individuelle et retourne son score et son statut.
        """
        result = {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "is_valid": False,
            "rejection_reason": None,
            "laplacian_variance": 0.0,
            "overexposed_ratio": 0.0,
            "underexposed_ratio": 0.0,
            "resolution": (0, 0)
        }

        if not os.path.exists(image_path):
            result["rejection_reason"] = "FILE_NOT_FOUND"
            return result

        # 1. Vérification d'intégrité de lecture
        try:
            # Test d'ouverture PIL pour vérifier les headers JPEG
            with Image.open(image_path) as pil_img:
                pil_img.verify()
        except Exception as e_pil:
            result["rejection_reason"] = f"CORRUPTED_HEADER: {str(e_pil)}"
            return result

        # Lecture OpenCV pour le traitement numérique
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            result["rejection_reason"] = "DECODE_FAILED"
            return result

        h, w = img_bgr.shape[:2]
        result["resolution"] = (w, h)

        if w < self.min_resolution[0] or h < self.min_resolution[1]:
            result["rejection_reason"] = f"RESOLUTION_TOO_LOW: {w}x{h}"
            return result

        # 2. Évaluation de l'exposition (Espace HSV)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        total_pixels = float(h * w)

        overexposed_pixels = np.sum(v_channel > 250)
        overexposed_ratio = float(overexposed_pixels / total_pixels)
        result["overexposed_ratio"] = round(overexposed_ratio, 4)

        underexposed_pixels = np.sum(v_channel < 15)
        underexposed_ratio = float(underexposed_pixels / total_pixels)
        result["underexposed_ratio"] = round(underexposed_ratio, 4)

        if overexposed_ratio > self.max_overexposed_ratio:
            result["rejection_reason"] = f"OVEREXPOSED_FLASH (ratio={overexposed_ratio:.2f})"
            return result

        if underexposed_ratio > self.max_underexposed_ratio:
            result["rejection_reason"] = f"UNDEREXPOSED_DARK (ratio={underexposed_ratio:.2f})"
            return result

        # 3. Évaluation du flou (Variance du Laplacien)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        result["laplacian_variance"] = round(laplacian_var, 2)

        if laplacian_var < self.min_laplacian_var:
            result["rejection_reason"] = f"BLURRY_IMAGE (var={laplacian_var:.1f} < {self.min_laplacian_var})"
            return result

        # Image conforme
        result["is_valid"] = True
        return result

    def audit_dataset_manifest(self, input_csv: str, output_report_csv: str) -> pd.DataFrame:
        """
        Scanne un fichier CSV contenant une colonne 'image_path' ou 'filepath' et génère le rapport qualité.
        """
        df = pd.read_csv(input_csv)
        path_col = "filepath" if "filepath" in df.columns else ("image_path" if "image_path" in df.columns else None)
        if path_col is None:
            raise ValueError("Le CSV doit contenir une colonne 'filepath' ou 'image_path'.")

        results = []
        for path in tqdm(df[path_col], desc="Audit Qualité des Images"):
            res = self.evaluate_image(path)
            results.append(res)

        report_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_report_csv), exist_ok=True)
        report_df.to_csv(output_report_csv, index=False)

        valid_count = report_df["is_valid"].sum()
        total_count = len(report_df)
        print(f"📊 Audit Qualité Terminé : {valid_count}/{total_count} images conformes ({(valid_count/total_count)*100:.1f}%).")
        print(f"📄 Rapport sauvegardé → {output_report_csv}")

        return report_df

if __name__ == "__main__":
    q_filter = CervicalImageQualityFilter()
    print("✅ Module de Filtrage Qualité Initialisé.")
