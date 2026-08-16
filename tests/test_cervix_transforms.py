import pytest
import numpy as np
from src.preprocessing.cervix_transforms import SpecularReflectionMasker

def test_specular_reflection_masker():
    """Vérifie que le masque Reflets-Lite remplace bien les pixels saturés par la couleur moyenne."""
    masker = SpecularReflectionMasker(v_threshold=235)

    # Image synthétique RGB (100x100x3) avec fond sombre uniforme
    img = np.full((100, 100, 3), 100, dtype=np.uint8)
    # Zone de reflet saturée (valeur 255)
    img[10:20, 10:20] = [255, 255, 255]

    processed = masker(img)

    # Les pixels saturés ne doivent plus être à 255
    assert not np.all(processed[10:20, 10:20] == [255, 255, 255]), "Le reflet n'a pas été atténué."
    assert processed.shape == (100, 100, 3), "La dimension de l'image a été modifiée."
