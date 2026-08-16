import torch
import torch.nn as nn
import timm

class IVALesionClassifierStage2(nn.Module):
    """
    Étape 2 (CADx) : Backbone vision (ConvNeXt-Base ou Swin-v2) multi-tâche.
    Tâche A : Éligibilité anatomique IVA (Type 1/2 vs Type 3 Invalide).
    Tâche B : Diagnostic Lésionnel IVA (Positif vs Négatif).
    """
    def __init__(
        self,
        backbone_name: str = "convnext_base",
        pretrained: bool = True,
        num_classes_eligibility: int = 3,
        num_classes_pathology: int = 2,
        drop_rate: float = 0.2
    ):
        super().__init__()
        self.backbone_name = backbone_name
        
        # Chargement du backbone timm pré-entraîné sur ImageNet
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0, # Feature extractor
            drop_rate=drop_rate
        )
        
        in_features = self.backbone.num_features
        
        # Tête de classification d'Éligibilité Anatomique (Type 1, Type 2, Type 3)
        self.head_eligibility = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate / 2),
            nn.Linear(512, num_classes_eligibility)
        )
        
        # Tête de classification de Diagnostic Lésionnel (Positif vs Négatif)
        self.head_pathology = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(drop_rate / 2),
            nn.Linear(512, num_classes_pathology)
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits_eligibility = self.head_eligibility(features)
        logits_pathology = self.head_pathology(features)
        
        return {
            "eligibility": logits_eligibility,
            "pathology": logits_pathology
        }

if __name__ == "__main__":
    model = IVALesionClassifierStage2(pretrained=False)
    dummy_input = torch.randn(2, 3, 384, 384)
    out = model(dummy_input)
    print(f"✅ Modèle Stage 2 initialisé. Logits éligibilité : {out['eligibility'].shape}, Logits pathologie : {out['pathology'].shape}")
