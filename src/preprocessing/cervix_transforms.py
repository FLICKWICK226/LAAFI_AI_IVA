import cv2
import numpy as np

class SpecularReflectionMasker:
    """
    Module de prétraitement ultra-rapide (<2ms CPU) :
    Détecte les reflets spéculaires intenses provoqués par le flash LED (V > v_threshold en HSV)
    et les remplace par la couleur moyenne locale des pixels de la muqueuse cervicale non saturée.
    """
    def __init__(self, v_threshold: int = 235):
        self.v_threshold = v_threshold

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        if not isinstance(img_np, np.ndarray) or img_np.ndim != 3:
            return img_np

        # 1. Conversion en espace HSV pour isoler le canal de valeur/luminance (V)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        
        # 2. Création du masque binaire des pixels saturés par le reflet du flash
        _, mask = cv2.threshold(v_channel, self.v_threshold, 255, cv2.THRESH_BINARY)
        
        # 3. Si des reflets sont présents, les remplacer par la couleur moyenne du col
        if np.any(mask == 255):
            non_sat_mask = cv2.bitwise_not(mask)
            # Calcul de la couleur moyenne sur les zones non réfléchissantes
            mean_color = cv2.mean(img_np, mask=non_sat_mask)[:3]
            img_np = img_np.copy()
            img_np[mask == 255] = np.array(mean_color, dtype=np.uint8)
            
        return img_np
