"""
model.py — KeeKee v3 Main Model
====================================
Gemma-inspired (nn/_transformer.py Transformer):
  - Weight-tied embedding ↔ lm_head (same matrix, saves params)
  - RoPE cache (local + global frequencies per attention type)
  - Per-block attention types from config.attention_types
  - `return_last_only` flag (decode only last token during AR generation)
  - Final logit soft-cap (Gemma2 optional)
  - init_cache() delegates to config
  - clean forward signature matches Gemma's Output namedtuple pattern
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from config import GPTConfig, AttentionType
from layers import RMSNorm
from block import Block
from rope import RoPECache
from kv_cache import Cache, clear_cache


# ─────────────────────────────────────────────
# Model Output  (mirrors Gemma's Output namedtuple)
# ─────────────────────────────────────────────
@dataclass
class ModelOutput:
    logits: torch.Tensor                   # [B, T, vocab] or [B, 1, vocab]
    cache: Optional[Cache] = None          # updated KV cache
    hidden_states: Optional[torch.Tensor] = None  # [B, T, D] before lm_head


# ─────────────────────────────────────────────
# KeeKee v3
# ─────────────────────────────────────────────
class KeeKee(nn.Module):
    """
    KeeKee v3 — Gemma-architecture-inspired language model.

    Key upgrades over v2:
      - GQA in every attention block (n_kv_head < n_head)
      - Alternating LOCAL_SLIDING / GLOBAL attention (5:1 ratio, Gemma3 style)
      - QK-Norm inside attention (Gemma3+)
      - Post-attention and post-FFW RMSNorms (Gemma2/3 stability)
      - Two RoPE frequency tables: local (base=10_000) + global (base=1_000_000)
      - Optional final logit soft-cap (Gemma2)
      - return_last_only mode for fast AR decoding (no wasted computation)
      - Explicit KV cache (passed in/out, no hidden state)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # ── Token embedding ──────────────────────────────────────────────
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # No wpe — RoPE handles position encoding entirely

        # ── Transformer blocks (one per layer, with per-block attn type) ─
        self.blocks = nn.ModuleList([
            Block(config, attn_type)
            for attn_type in config.attention_types
        ])

        # ── Final norm + LM head ─────────────────────────────────────────
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: lm_head shares embedding matrix (Gemma / GPT-2 convention)
        # Saves ~vocab_size × n_embd params and improves token consistency
        self.lm_head.weight = self.wte.weight

        # ── RoPE cache: separate cos/sin for local vs global layers ──────
        self.rope_cache = RoPECache(
            head_dim=config.head_dim,
            max_seq_len=config.block_size,
            local_base_frequency=config.rope_base_frequency,
            global_base_frequency=1_000_000,          # Gemma3 global layers
            local_scale_factor=config.rope_scale_factor,
            global_scale_factor=8.0,                  # Gemma3 global_scale_factor
            device="cpu",                             # moved to device in .to()
        )

        self._init_weights()
        self.gradient_checkpointing = False   # enable via model.gradient_checkpointing = True

    # ── Weight initialisation ────────────────────────────────────────────
    def _init_weights(self):
        """Gemma-style normal(0, 0.02) init for all linear and embedding layers."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def to(self, *args, **kwargs):
        """Override to() so RoPE cache moves with the model."""
        result = super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self.rope_cache.to(device)
        return result

    # ── Cache initialisation ─────────────────────────────────────────────
    def init_cache(self, batch_size: int) -> Cache:
        """Allocate a fresh KV cache for all layers."""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        return self.config.init_cache(batch_size, device=str(device), dtype=dtype)

    # ── Forward pass ─────────────────────────────────────────────────────
    def forward(
        self,
        idx: torch.Tensor,                  # [B, T] token ids
        cache: Optional[Cache] = None,      # KV cache (None = no cache)
        return_last_only: bool = False,     # True → only return logits for last token
        return_hidden_states: bool = False, # True → also return pre-logit hidden states
    ) -> ModelOutput:
        """
        Forward pass.

        Args:
            idx:               [B, T] int64 token ids
            cache:             optional KV cache dict (from init_cache or prior forward)
            return_last_only:  if True, only compute logits for idx[:, -1, :]
                               (Gemma pattern: massive saving during AR decoding)
            return_hidden_states: if True, return the final hidden states too

        Returns:
            ModelOutput with .logits, .cache, .hidden_states
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} > block_size {self.config.block_size}"
        )

        # ── 1. Token embeddings (no position added — RoPE is in attention) ─
        x = self.wte(idx)   # [B, T, D]

        # Embedding scaling: √n_embd  (Gemma's Embedder.encode scaling)
        x = x * (self.config.n_embd ** 0.5)

        # ── 2. Determine current cache position ──────────────────────────
        if cache is not None:
            # How far into the sequence are we?
            layer0_cache = cache["layer_0"]
            cache_end_idx = int(layer0_cache["end_index"][0].item())
        else:
            cache_end_idx = 0

        # ── 3. Run transformer blocks ────────────────────────────────────
        for i, block in enumerate(self.blocks):
            layer_key = f"layer_{i}"
            layer_cache = cache[layer_key] if cache is not None else None

            is_global = (block.attn_type == AttentionType.GLOBAL)
            freqs_cos, freqs_sin = self.rope_cache.get(T, is_global=is_global)

            if self.gradient_checkpointing and self.training and cache is None:
                # Wrap block in gradient checkpoint: recomputes activations on backward
                # pass instead of storing them → ~50% less activation memory
                def make_block_fn(blk, fc, fs):
                    def block_fn(x_):
                        out, _ = blk(x_, fc, fs, cache=None, cache_end_idx=0)
                        return out
                    return block_fn
                x = grad_checkpoint(make_block_fn(block, freqs_cos, freqs_sin), x,
                                    use_reentrant=False)
                updated_layer_cache = None
            else:
                x, updated_layer_cache = block(
                    x, freqs_cos, freqs_sin,
                    cache=layer_cache,
                    cache_end_idx=cache_end_idx,
                )

            # Write back updated layer cache
            if cache is not None and updated_layer_cache is not None:
                cache[layer_key] = updated_layer_cache

        # ── 4. Final norm ────────────────────────────────────────────────
        x = self.norm_f(x)   # [B, T, D]

        # ── 5. Optionally return last token only (AR decoding) ───────────
        #    Gemma's return_last_only: avoids projecting all T tokens to vocab
        hidden = x[:, -1:, :] if return_last_only else x   # [B, 1, D] or [B, T, D]

        # ── 6. Logit projection (weight-tied with embedding) ─────────────
        logits = self.lm_head(hidden)   # [B, 1, V] or [B, T, V]

        # ── 7. Final logit soft-cap (Gemma2 optional) ────────────────────
        cap = self.config.final_logit_softcap
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap

        return ModelOutput(
            logits=logits,
            cache=cache,
            hidden_states=x if return_hidden_states else None,
        )

    # ── Param counting ───────────────────────────────────────────────────
    def num_params(self, exclude_embedding: bool = True) -> int:
        """
        Count trainable parameters.
        exclude_embedding=True: don't count the shared embedding/lm_head matrix
        (matches how GPT-2 paper reports param count).
        """
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embedding:
            n -= self.wte.weight.numel()
        return n
