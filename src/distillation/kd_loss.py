"""
Module de perte pour la Distillation Hybride Binaire (Teacher ➔ Student).
Combine Soft-BCE (Distillation de connaissances), Hard Loss Clinique (Asymmetric Focal Loss / BCE)
et Attention Transfer Spatiale inter-couches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.asymmetric_loss import AsymmetricFocalLoss

class BinaryHybridKDLoss(nn.Module):
    """
    Distillation Hybride Binaire :
    1. Soft-BCE Logits (Distillation de connaissances avec Température T)
    2. Hard Loss Clinique (AsymmetricFocalLoss ou BCE)
    3. Attention Transfer (Alignement L2 des cartes d'activation spatiales)
    """
    def __init__(
        self,
        alpha_kd: float = 0.6,
        beta_attn: float = 50.0,
        temperature: float = 4.0,
        use_asymmetric_hard_loss: bool = True
    ):
        super().__init__()
        self.alpha_kd = alpha_kd
        self.beta_attn = beta_attn
        self.T = temperature
        self.use_asymmetric_hard_loss = use_asymmetric_hard_loss
        
        if self.use_asymmetric_hard_loss:
            self.hard_loss_fn = AsymmetricFocalLoss(gamma_pos=1.0, gamma_neg=4.0)
        else:
            self.hard_loss_fn = nn.BCEWithLogitsLoss()

    def _get_spatial_attention_map(self, feature_map: torch.Tensor, target_spatial_size: tuple) -> torch.Tensor:
        """
        Calcule la carte d'attention spatiale L2 agrégée sur les canaux.
        Resample à target_spatial_size (H, W) et normalise L2.
        """
        if feature_map is None:
            return None
        # Spatial pooling somme des carrés des activations des canaux: [B, 1, H, W]
        spatial_map = feature_map.pow(2).mean(dim=1, keepdim=True)
        # Redimensionnement adaptatif vers la taille cible
        resampled_map = F.adaptive_avg_pool2d(spatial_map, target_spatial_size)
        # Normalisation L2 par échantillon dans le batch
        flat_map = resampled_map.view(resampled_map.size(0), -1)
        norm_map = F.normalize(flat_map, p=2, dim=1)
        return norm_map

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_feats: torch.Tensor = None,
        teacher_feats: torch.Tensor = None,
        targets: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calcul de la perte globale de distillation.
        """
        s_logits = student_logits.view(-1)
        t_logits = teacher_logits.view(-1)

        # 1. Soft-BCE Loss avec Température T
        teacher_probs = torch.sigmoid(t_logits / self.T)
        loss_kd = F.binary_cross_entropy_with_logits(
            s_logits / self.T, teacher_probs
        ) * (self.T ** 2)

        # 2. Hard Target Loss Clinique (Vérité terrain)
        if targets is not None:
            if self.use_asymmetric_hard_loss:
                loss_hard = self.hard_loss_fn(s_logits, targets)
            else:
                loss_hard = F.binary_cross_entropy_with_logits(s_logits, targets.float().view(-1))
        else:
            loss_hard = torch.tensor(0.0, device=student_logits.device)

        # 3. Spatial Attention Transfer
        loss_attn = torch.tensor(0.0, device=student_logits.device)
        if student_feats is not None and teacher_feats is not None:
            target_spatial_shape = student_feats.shape[2:]
            s_attn = self._get_spatial_attention_map(student_feats, target_spatial_shape)
            t_attn = self._get_spatial_attention_map(teacher_feats, target_spatial_shape)
            if s_attn is not None and t_attn is not None:
                loss_attn = F.mse_loss(s_attn, t_attn)

        total_loss = (1.0 - self.alpha_kd) * loss_hard + self.alpha_kd * loss_kd + self.beta_attn * loss_attn

        return total_loss
