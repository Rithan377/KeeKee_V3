"""
layers.py — KeeKee v3 Primitive Layers
==========================================
Gemma-inspired (nn/_layers.py):
  - RMSNorm: zero-init scale (so initial scale = 1 + 0 = 1),
             matching Gemma's `normed * (1 + scale)` pattern.
  - No bias by default (Gemma uses bias=False everywhere).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# RMSNorm  (mirrors Gemma nn/_layers.py RMSNorm)
# Key difference from v2: scale is initialized to ZERO,
# so the initial transform is x * (1 + 0) = x (identity).
# This is more stable than initializing to ones.
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Gemma pattern:
        normed = x * rsqrt(mean(x²) + ε)
        output = normed * (1 + scale)   ← scale init'd to 0 → identity at start
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Zero-init: initial output = x * (1+0) = x  (matches Gemma)
        self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS norm
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # Apply learnable scale as (1 + scale) — zero-init means starts as identity
        return norm * (1.0 + self.scale)


# ─────────────────────────────────────────────
# Linear — no-bias wrapper (Gemma style)
# ─────────────────────────────────────────────
class Linear(nn.Linear):
    """Linear layer with bias=False by default (Gemma convention)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        # Gemma-style initialisation: normal(0, 0.02)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
