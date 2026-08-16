"""
Module d'architecture Student (MobileNetV4-Small) pour le déploiement Edge AI.
Intègre la capture dynamique des cartes de caractéristiques intermédiaires pour l'Attention Transfer.
"""

import torch
import torch.nn as nn
import timm

class MobileNetV4Student(nn.Module):
    """
    Modèle Student optimisé pour l'inférence sur smartphone Android (Edge AI).
    Backbone : mobilenetv4_conv_small (ou mobilenetv4_small)
    Sortie : Logit binaire [B, 1] (Diagnostic IVA Positif vs Négatif)
    """
    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small",
        pretrained: bool = True,
        drop_rate: float = 0.2,
        num_classes: int = 1
    ):
        super().__init__()
        self.backbone_name = backbone_name
        
        # Création du backbone MobileNetV4 via timm
        try:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=True,
                drop_rate=drop_rate
            )
        except Exception:
            # Fallback vers mobilenetv3_small_100 si mobilenetv4_conv_small n'est pas dispo dans la version timm installée
            print(f"⚠️ {backbone_name} non disponible. Fallback vers mobilenetv3_small_100")
            self.backbone_name = "mobilenetv3_small_100"
            self.backbone = timm.create_model(
                "mobilenetv3_small_100",
                pretrained=pretrained,
                features_only=True,
                drop_rate=drop_rate
            )
            
        feature_info = self.backbone.feature_info
        in_features = feature_info[-1]['num_chs']

        # Pooling global + tête de classification binaire
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(in_features, num_classes)
        )

        self._last_feature_map = None

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        # La dernière carte de caractéristiques spatiale
        self._last_feature_map = feats[-1]
        
        pooled = self.global_pool(self._last_feature_map)
        logits = self.classifier(pooled)
        return logits

    def get_last_feature_map(self) -> torch.Tensor:
        """Retourne la dernière carte de caractéristiques enregistrée pour Attention Transfer."""
        return self._last_feature_map

if __name__ == "__main__":
    model = MobileNetV4Student(pretrained=False)
    dummy_input = torch.randn(2, 3, 384, 384)
    logits = model(dummy_input)
    feat_map = model.get_last_feature_map()
    print(f"✅ Student MobileNetV4 initialisé. Logits : {logits.shape}, Feature Map : {feat_map.shape}")
