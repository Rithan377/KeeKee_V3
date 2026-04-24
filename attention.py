"""
attention.py — Multi-Head Attention (GQA + Sliding Window + QK Norm)
======================================================================
Gemma-inspired (nn/_modules.py Attention class):
  - GQA (Grouped Query Attention): n_kv_head < n_head
  - QK-Norm (RMSNorm on Q and K before attention, Gemma3+)
  - Sliding window mask for LOCAL_SLIDING attention type
  - Logit soft-cap: tanh(logits/cap)*cap (Gemma2 style)
  - Flash Attention via F.scaled_dot_product_attention
  - KV cache integration (Gemma-style dict cache)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from config import GPTConfig, AttentionType
from layers import RMSNorm, Linear
from rope import apply_rope
from kv_cache import LayerCache, update_cache


# ─────────────────────────────────────────────
# Sliding Window Mask helper
# (mirrors Gemma's create_sliding_mask)
# ─────────────────────────────────────────────
def create_sliding_mask(
    seq_len: int,
    cache_len: int,
    cache_end_idx: int,
    window_size: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Creates a boolean sliding window mask [T, cache_len].
    Token at position `pos` can only attend to positions within
    [pos - window_size, pos + window_size) in the cache.
    """
    # Query positions: [seq_len]
    q_pos = torch.arange(cache_end_idx, cache_end_idx + seq_len, device=device)
    # Key positions: [cache_len]
    k_pos = torch.arange(cache_len, device=device)

    # Broadcast: [seq_len, cache_len]
    diff = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)
    mask = (diff >= 0) & (diff < window_size)
    return mask   # [T, cache_len]


# ─────────────────────────────────────────────
# Attention Module
# ─────────────────────────────────────────────
class Attention(nn.Module):
    """
    Multi-Head Attention with Gemma-inspired features:
      - Grouped Query Attention (GQA) when n_kv_head < n_head
      - QK-Norm (RMSNorm on Q and K)
      - Soft logit cap (tanh capping, Gemma2 style)
      - Sliding window mask for local attention
      - KV cache for autoregressive inference
      - Flash Attention via PyTorch 2.0+ fused kernel
    """

    def __init__(self, config: GPTConfig, attn_type: AttentionType):
        super().__init__()
        self.config = config
        self.attn_type = attn_type
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.n_groups = config.n_head // config.n_kv_head  # GQA groups

        D = config.n_embd
        H = config.n_head
        KV = config.n_kv_head
        Hd = config.head_dim

        # Q projects to full n_head; K,V project to n_kv_head (GQA)
        self.wq = Linear(D, H * Hd)
        self.wk = Linear(D, KV * Hd)
        self.wv = Linear(D, KV * Hd)
        self.wo = Linear(H * Hd, D)

        # QK-Norm (Gemma3+): normalise Q and K before dot-product
        self.qk_norm = config.use_qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(Hd)
            self.k_norm = RMSNorm(Hd)

        self.attn_drop = nn.Dropout(config.dropout)
        self.query_scalar = config.query_pre_attn_scalar()   # 1/√head_dim

    def forward(
        self,
        x: torch.Tensor,                        # [B, T, D]
        freqs_cos: torch.Tensor,                # [T, head_dim//2]
        freqs_sin: torch.Tensor,
        cache: Optional[LayerCache] = None,     # KV cache dict
        cache_end_idx: int = 0,                 # how many tokens already cached
    ) -> tuple[torch.Tensor, Optional[LayerCache]]:
        """
        Forward pass.

        Returns:
            output: [B, T, D]
            updated_cache: LayerCache (or None if cache is None)
        """
        B, T, D = x.shape
        H, KV, Hd = self.n_head, self.n_kv_head, self.head_dim

        # ── 1. Project Q, K, V ───────────────────────────────────────────
        q = self.wq(x).view(B, T, H, Hd).transpose(1, 2)    # [B, H, T, Hd]
        k = self.wk(x).view(B, T, KV, Hd).transpose(1, 2)   # [B, KV, T, Hd]
        v = self.wv(x).view(B, T, KV, Hd).transpose(1, 2)   # [B, KV, T, Hd]

        # ── 2. QK-Norm (Gemma3) ─────────────────────────────────────────
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ── 3. Apply RoPE ────────────────────────────────────────────────
        q, k = apply_rope(q, k, freqs_cos, freqs_sin)

        # ── 4. Scale Q (Gemma: query_pre_attn_scalar = 1/√head_dim) ─────
        q = q * self.query_scalar

        # ── 5. KV Cache update ───────────────────────────────────────────
        if cache is not None:
            # Permute for cache format: [B, T, KV, Hd]
            k_to_cache = k.transpose(1, 2)
            v_to_cache = v.transpose(1, 2)
            k_full, v_full, cache = update_cache(cache, k_to_cache, v_to_cache)
            # [B, cache_size, KV, Hd] → [B, KV, cache_size, Hd]
            k = k_full.transpose(1, 2)
            v = v_full.transpose(1, 2)

        # ── 6. GQA: expand K, V to match Q heads ────────────────────────
        if self.n_groups > 1:
            # Repeat K, V heads to match n_head
            k = k.repeat_interleave(self.n_groups, dim=1)   # [B, H, S, Hd]
            v = v.repeat_interleave(self.n_groups, dim=1)

        # ── 7. Soft Logit Cap (Gemma2) ───────────────────────────────────
        # We compute logits manually only if we need the cap; otherwise
        # fall through to flash attention (no explicit cap supported in fused kernel)
        cap = self.config.attn_logits_soft_cap
        use_manual = (cap is not None) or (
            self.attn_type == AttentionType.LOCAL_SLIDING and cache is not None
        )

        if use_manual:
            # Manual attention: compute logits explicitly
            S = k.shape[2]  # key/cache length

            # Logits: [B, H, T, S]
            logits = torch.matmul(q, k.transpose(-1, -2))

            # Soft cap
            if cap is not None:
                logits = torch.tanh(logits / cap) * cap

            # Causal mask
            attn_mask = torch.ones(T, S, device=x.device, dtype=torch.bool).tril()

            # Sliding window mask (LOCAL_SLIDING layers)
            if self.attn_type == AttentionType.LOCAL_SLIDING:
                sw_mask = create_sliding_mask(
                    T, S, cache_end_idx,
                    self.config.sliding_window_size, x.device
                )
                attn_mask = attn_mask & sw_mask

            logits = logits.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            probs = F.softmax(logits, dim=-1)
            probs = self.attn_drop(probs)
            y = torch.matmul(probs, v)   # [B, H, T, Hd]

        else:
            # Flash Attention (PyTorch 2.0+ fused kernel) — no explicit mask needed
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
                scale=1.0,   # already scaled q above
            )   # [B, H, T, Hd]

        # ── 8. Output projection ─────────────────────────────────────────
        y = y.transpose(1, 2).contiguous().view(B, T, H * Hd)  # [B, T, D]
        out = self.wo(y)

        return out, cache
