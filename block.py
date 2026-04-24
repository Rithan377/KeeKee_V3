"""
block.py — Transformer Block
==============================
Gemma-inspired (nn/_modules.py Block):
  - Pre-attention norm (always)
  - Optional post-attention norm (Gemma2/3 stability trick)
  - Pre-FFW norm (always)
  - Optional post-FFW norm
  - Attention type (GLOBAL or LOCAL_SLIDING) passed at construction
  - Residual connections around both sub-layers
"""

import torch
import torch.nn as nn
from typing import Optional

from config import GPTConfig, AttentionType
from layers import RMSNorm
from attention import Attention
from mlp import GeGLU
from kv_cache import LayerCache


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────
class Block(nn.Module):
    """
    Single Transformer block.

    Structure (Gemma-style pre-norm with optional post-norms):

        x
        ├─ pre_attn_norm(x) → Attention → [post_attn_norm] → + x  (residual)
        └─ pre_ffw_norm(above) → GeGLU → [post_ffw_norm] → + above  (residual)

    Post-norms (use_post_attn_norm, use_post_ffw_norm) are Gemma2/3 additions
    that provide extra training stability at the cost of 2 extra RMSNorm params.
    """

    def __init__(self, config: GPTConfig, attn_type: AttentionType):
        super().__init__()
        self.attn_type = attn_type

        # Pre-norms (always present)
        self.pre_attn_norm = RMSNorm(config.n_embd)
        self.pre_ffw_norm = RMSNorm(config.n_embd)

        # Post-norms (Gemma2/3 optional)
        self.post_attn_norm = (
            RMSNorm(config.n_embd) if config.use_post_attn_norm else nn.Identity()
        )
        self.post_ffw_norm = (
            RMSNorm(config.n_embd) if config.use_post_ffw_norm else nn.Identity()
        )

        # Sub-layers
        self.attn = Attention(config, attn_type)
        self.mlp = GeGLU(config)

    def forward(
        self,
        x: torch.Tensor,                     # [B, T, D]
        freqs_cos: torch.Tensor,             # [T, head_dim//2]
        freqs_sin: torch.Tensor,
        cache: Optional[LayerCache] = None,
        cache_end_idx: int = 0,
    ) -> tuple[torch.Tensor, Optional[LayerCache]]:
        """
        Forward pass through one transformer block.

        Returns:
            x:            [B, T, D]
            updated_cache: LayerCache or None
        """
        # ── Attention sub-layer ──────────────────────────────────────────
        attn_out, cache = self.attn(
            self.pre_attn_norm(x),
            freqs_cos, freqs_sin,
            cache=cache,
            cache_end_idx=cache_end_idx,
        )
        x = x + self.post_attn_norm(attn_out)   # residual + optional post-norm

        # ── FFW sub-layer ────────────────────────────────────────────────
        ffw_out = self.mlp(self.pre_ffw_norm(x))
        x = x + self.post_ffw_norm(ffw_out)     # residual + optional post-norm

        return x, cache
