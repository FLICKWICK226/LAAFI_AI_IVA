import os
import cv2
import numpy as np
import torch

class JSCDetectorStage1:
    """
    Étape 1 (CADe) : Détecteur d'Objet YOLOv8/v11-Det pour la Zone de Jonction Squamo-Columnaire (JSC).
    Élimine > 70% de l'arrière-plan (spéculum, vagin, métal) pour contrer le Shortcut Learning.
    """
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        self.model = None
        
        if model_path and os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                print(f"✅ Modèle Stage 1 YOLO-Det chargé depuis : {model_path}")
            except Exception as e:
                print(f"⚠️ Impossible de charger YOLO depuis {model_path}: {e}")

    def crop_jsc(self, image: np.ndarray, target_size: tuple = (384, 384)) -> np.ndarray:
        """
        Détecte la JSC et découpe la ROI. Si aucune détection n'est au-dessus du seuil,
        effectue un rognage central de sécurité (Center Crop).
        """
        h, w, c = image.shape
        
        if self.model is not None:
            results = self.model(image, conf=self.conf_threshold, verbose=False)
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Extraire la boîte englobante avec la plus haute confiance
                boxes = results[0].boxes
                best_box = boxes[torch.argmax(boxes.conf)].xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = best_box
                
                # Marge de sécurité 10%
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                crop = image[y1:y2, x1:x2]
                return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

        # Fallback : Center Crop 70% si pas de détection YOLO
        crop_h, crop_w = int(h * 0.7), int(w * 0.7)
        start_y = (h - crop_h) // 2
        start_x = (w - crop_w) // 2
        crop = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
        return cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)

if __name__ == "__main__":
    detector = JSCDetectorStage1()
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    cropped = detector.crop_jsc(dummy_img)
    print(f"✂️ Crop JSC généré avec taille : {cropped.shape}")
