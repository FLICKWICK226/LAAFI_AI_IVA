import torch
import torch.nn as nn
import timm

class IVALesionClassifierStage2(nn.Module):
    """
    Étape 2 (CADx) : Backbone vision ConvNeXt unifié (Single-Head 3-Classes).
    Classe 0 : Type 1 (Col Normal / ZT Entièrement Visible -> Dépistage Négatif)
    Classe 1 : Type 2 (Col Lésionnel / ZT Partiellement Visible -> Dépistage Positif / Référer)
    Classe 2 : Type 3 (Col Non Visualisable / ZT Non Visible -> Examen Non-Éligible / Incomplet)
    """
    def __init__(
        self,
        backbone_name: str = "convnext_small",
        pretrained: bool = True,
        num_classes: int = 3,
        drop_rate: float = 0.2
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Chargement du backbone timm pré-entraîné sur ImageNet
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0, # Feature extractor
            drop_rate=drop_rate
        )
        
        in_features = self.backbone.num_features
        
        # Tête de classification unifiée à 3 classes anatomiques / décisionnelles
        self.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retourne directement les logits de classification de forme [batch_size, num_classes].
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

if __name__ == "__main__":
    model = IVALesionClassifierStage2(pretrained=False, num_classes=3)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print(f"[OK] Modele Stage 2 Unifie initialise. Logits shape : {out.shape} (attendu: [2, 3])")

