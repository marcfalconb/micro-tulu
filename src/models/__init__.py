"""Frontier architectural modules inspired by Kimi K3 and OLMo 3."""

from src.models.activations import SiTUGLU, situglu_activation
from src.models.attn_res import BlockAttentionResiduals
from src.models.latent_moe import StableLatentMoE
from src.models.router import QuantileBalancingRouter

__all__ = [
    "SiTUGLU",
    "situglu_activation",
    "BlockAttentionResiduals",
    "StableLatentMoE",
    "QuantileBalancingRouter",
]
