"""
mlp.py — Feed-Forward Network (GeGLU / SwiGLU)
=================================================
Gemma-inspired (nn/_modules.py FeedForward):
  - Uses GeGLU gating: GELU(gate) * value, then project back
  - Two separate gate/value projection matrices (like Gemma's gating_einsum split)
  - Hidden dim follows Gemma sizing: 2/3 * 4 * n_embd rounded to mult of 64
  - No bias (Gemma convention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from layers import Linear


# ─────────────────────────────────────────────
# GeGLU MLP  (matches Gemma's FeedForward module)
# ─────────────────────────────────────────────
class GeGLU(nn.Module):
    """
    Gated Feed-Forward Network with GELU activation (GeGLU).

    Gemma's FeedForward:
        gate, value = split(gating_einsum(x), 2)
        hidden = gelu(gate) * value
        output = linear(hidden)

    This is equivalent to SwiGLU but uses GELU instead of SiLU.
    Empirically ~10% better loss than plain ReLU FFW at same param count.

    Architecture:
        x [B,T,D] → [gate [B,T,H], value [B,T,H]] → gelu(gate)*value [B,T,H] → [B,T,D]
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        H = config.ffw_hidden_dim   # computed in config (2/3 * 4 * n_embd, mult of 64)
        D = config.n_embd

        # Gate projection (Gemma: gating_einsum[:, 0, :])
        self.w_gate = Linear(D, H)
        # Value projection (Gemma: gating_einsum[:, 1, :])
        self.w_val = Linear(D, H)
        # Output projection (Gemma: linear)
        self.w_out = Linear(H, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gated activation: GELU(gate) ⊙ value  (GeGLU)
        gate = F.gelu(self.w_gate(x), approximate="tanh")
        val = self.w_val(x)
        hidden = gate * val           # element-wise gate
        return self.w_out(hidden)


# ─────────────────────────────────────────────
# SwiGLU variant (uses SiLU instead of GELU)
# kept for experimentation / drop-in swap
# ─────────────────────────────────────────────
class SwiGLU(nn.Module):
    """
    SwiGLU variant: SiLU(gate) * value.
    Slightly faster than GeGLU; used in LLaMA family.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        H = config.ffw_hidden_dim
        D = config.n_embd
        self.w_gate = Linear(D, H)
        self.w_val = Linear(D, H)
        self.w_out = Linear(H, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(F.silu(self.w_gate(x)) * self.w_val(x))
