"""
rope.py — Rotary Position Embeddings (RoPE)
============================================
Gemma-inspired (math/_positional_embeddings.py):
  - Separate base_frequency per attention type (local vs global)
  - scale_factor support for long-context extrapolation (Gemma3 global_scale_factor=8.0)
  - rope_proportion: partial RoPE (apply to first N% of head dims, rest unrotated)
"""

import torch
from typing import Optional


# ─────────────────────────────────────────────
# Precompute RoPE frequencies
# (matches Gemma's apply_rope logic)
# ─────────────────────────────────────────────
def precompute_rope_freqs(
    head_dim: int,
    seq_len: int,
    base_frequency: int = 10_000,
    scale_factor: float = 1.0,
    rope_proportion: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Precompute the (cos, sin) rotation components for RoPE.

    Args:
        head_dim:        Dimension per head (H).
        seq_len:         Maximum sequence length.
        base_frequency:  θ base (10_000 for local, 1_000_000 for Gemma3 global).
        scale_factor:    > 1.0 slows rotation → allows longer context extrapolation.
        rope_proportion: Fraction of head_dim to rotate (1.0 = all dims, Gemma default).
        device:          Torch device.

    Returns:
        freqs_cos, freqs_sin: both [seq_len, head_dim]
    """
    rope_dims = int(rope_proportion * head_dim)      # dims to rotate
    nope_dims = head_dim // 2 - rope_dims // 2       # dims left as-is (padded)

    # Compute inverse frequencies: θ_i = 1 / (base ** (2i / head_dim))
    exponents = torch.arange(0, rope_dims, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base_frequency ** (exponents / head_dim))

    # Position indices
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Outer product: [seq_len, rope_dims//2]
    sinusoid = torch.outer(positions, inv_freq)

    # Apply scale_factor (Gemma long-context trick: divides the angle)
    if scale_factor != 1.0:
        sinusoid = sinusoid / scale_factor

    # [seq_len, head_dim//2] — pad nope dims with 0 (no rotation)
    if nope_dims > 0:
        sinusoid = torch.cat([sinusoid, torch.zeros(seq_len, nope_dims, device=device)], dim=-1)

    freqs_cos = torch.cos(sinusoid)   # [seq_len, head_dim//2]
    freqs_sin = torch.sin(sinusoid)   # [seq_len, head_dim//2]

    return freqs_cos, freqs_sin


# ─────────────────────────────────────────────
# Apply RoPE to a single tensor
# ─────────────────────────────────────────────
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Split last dim in half and rotate:  [-x2, x1]
    Equivalent to Gemma's complex-multiply approach but stays real-valued.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.

    Args:
        q, k:        [B, n_head, T, head_dim]
        freqs_cos:   [T, head_dim//2]  (precomputed)
        freqs_sin:   [T, head_dim//2]

    Returns:
        q_rot, k_rot: same shape as inputs
    """
    # Reshape freqs for broadcast: [1, 1, T, head_dim//2]
    cos = freqs_cos.unsqueeze(0).unsqueeze(0)   # [1, 1, T, head_dim//2]
    sin = freqs_sin.unsqueeze(0).unsqueeze(0)

    # Split head_dim in half for rotation
    def rotate(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)
        return rotated.type_as(x)

    return rotate(q), rotate(k)


# ─────────────────────────────────────────────
# RoPE cache class — stores precomputed freqs
# for both local and global attention types
# ─────────────────────────────────────────────
class RoPECache:
    """
    Precomputed RoPE cos/sin tables for local and global attention layers.
    Mirrors Gemma's pattern: local uses base=10_000, global uses base=1_000_000
    with scale_factor=8.0 for long-range context.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        local_base_frequency: int = 10_000,
        global_base_frequency: int = 1_000_000,
        local_scale_factor: float = 1.0,
        global_scale_factor: float = 8.0,    # Gemma3 default for global layers
        device: str = "cpu",
    ):
        cos_local, sin_local = precompute_rope_freqs(
            head_dim, max_seq_len, local_base_frequency, local_scale_factor, device=device
        )
        cos_global, sin_global = precompute_rope_freqs(
            head_dim, max_seq_len, global_base_frequency, global_scale_factor, device=device
        )
        # Store on instance (will be moved to device with model)
        self.cos_local = cos_local
        self.sin_local = sin_local
        self.cos_global = cos_global
        self.sin_global = sin_global

    def get(self, seq_len: int, is_global: bool):
        """Return (cos, sin) sliced to `seq_len`."""
        if is_global:
            return self.cos_global[:seq_len], self.sin_global[:seq_len]
        return self.cos_local[:seq_len], self.sin_local[:seq_len]

    def to(self, device):
        self.cos_local = self.cos_local.to(device)
        self.sin_local = self.sin_local.to(device)
        self.cos_global = self.cos_global.to(device)
        self.sin_global = self.sin_global.to(device)
        return self
