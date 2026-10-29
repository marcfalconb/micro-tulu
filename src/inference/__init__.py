"""Inference accelerators: Speculative Decoding and Advanced Sampling."""

from src.inference.sampling import apply_min_p, apply_top_k_top_p, sample_next_token
from src.inference.speculative_decoding import SpeculativeDecoder, speculative_step

__all__ = [
    "apply_min_p",
    "apply_top_k_top_p",
    "sample_next_token",
    "speculative_step",
    "SpeculativeDecoder",
]
