"""
config.py — KeeKee v3 Configuration
======================================
Includes MODEL_150M production preset (150M params, BPE tokenizer, 2048 ctx).
======================================
Gemma-inspired: frozen dataclass, attention type pattern (local/global),
computed properties for head_dim and query_pre_attn_scalar, and init_cache().
"""

import math
import enum
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Attention Type Enum  (from Gemma nn/_modules.py)
# ─────────────────────────────────────────────
class AttentionType(enum.Enum):
    GLOBAL = "global"
    LOCAL_SLIDING = "local_sliding"


# ─────────────────────────────────────────────
# Normalization strategy  (from Gemma nn/_config.py)
# ─────────────────────────────────────────────
class QueryPreAttnNorm(enum.Enum):
    BY_ONE_OVER_SQRT_HEAD_DIM = "sqrt_head_dim"          # 1/√head_dim
    BY_ONE_OVER_SQRT_EMBED_DIV_HEADS = "sqrt_embed_heads"  # 1/√(embed//heads)


# ─────────────────────────────────────────────
# Attention Layer Pattern helper  (from Gemma nn/_config.py)
# ─────────────────────────────────────────────
def make_attention_pattern(
    pattern: tuple[AttentionType, ...],
    num_layers: int,
) -> tuple[AttentionType, ...]:
    """
    Tile `pattern` across `num_layers`.
    e.g. (LOCAL, LOCAL, LOCAL, LOCAL, LOCAL, GLOBAL) × n_layers
    """
    base = pattern * (num_layers // len(pattern))
    extra = pattern[: num_layers % len(pattern)]
    return tuple(base) + extra


# Gemma3-style: 5 local sliding + 1 global, repeated
GEMMA3_ATTENTION_PATTERN = (
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.LOCAL_SLIDING,
    AttentionType.GLOBAL,
)


# ─────────────────────────────────────────────
# GPTConfig  (Gemma-style frozen dataclass with computed properties)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class GPTConfig:
    # Core dimensions
    vocab_size: int = 512           # set at runtime from dataset
    n_layer: int = 10
    n_embd: int = 320              # embed_dim (D)
    n_head: int = 8                # query heads
    n_kv_head: int = 2             # key/value heads (GQA; set == n_head for MHA)
    block_size: int = 256          # max sequence length

    # FFW
    hidden_dim: Optional[int] = None  # if None, computed as 2/3 * 4 * n_embd

    # Attention behaviour
    attention_pattern: tuple = GEMMA3_ATTENTION_PATTERN   # per-block pattern
    sliding_window_size: int = 64                          # for LOCAL_SLIDING blocks

    # Normalization
    use_post_attn_norm: bool = True   # Gemma2/3 style: extra RMSNorm after attn output
    use_post_ffw_norm: bool = True    # Gemma2/3 style: extra RMSNorm after FFW output
    use_qk_norm: bool = True          # QK-norm inside attention (Gemma3+)
    query_pre_attn_norm: QueryPreAttnNorm = QueryPreAttnNorm.BY_ONE_OVER_SQRT_HEAD_DIM

    # Logit capping
    attn_logits_soft_cap: Optional[float] = 50.0   # Gemma2 style (None to disable)
    final_logit_softcap: Optional[float] = None     # Gemma2 style (None to disable)

    # RoPE
    rope_base_frequency: int = 10_000
    rope_scale_factor: float = 1.0    # > 1.0 for long-context extrapolation

    # Training
    dropout: float = 0.0        # 0 for inference, 0.1 for training

    # ── Computed properties ──────────────────────────────────────────────

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.n_embd // self.n_head

    @property
    def num_layers(self) -> int:
        return self.n_layer

    @property
    def ffw_hidden_dim(self) -> int:
        """Gemma-style: 2/3 * 4 * n_embd, rounded to multiple of 64."""
        if self.hidden_dim is not None:
            return self.hidden_dim
        h = int(2 / 3 * 4 * self.n_embd)
        return (h + 63) // 64 * 64

    @property
    def attention_types(self) -> tuple[AttentionType, ...]:
        """Full per-layer attention type list."""
        return make_attention_pattern(self.attention_pattern, self.n_layer)

    def query_pre_attn_scalar(self) -> float:
        """Returns the pre-attention scaling factor for queries."""
        if self.query_pre_attn_norm == QueryPreAttnNorm.BY_ONE_OVER_SQRT_EMBED_DIV_HEADS:
            return (self.n_embd // self.n_head) ** -0.5
        return self.head_dim ** -0.5   # default: 1/√head_dim

    def init_cache(self, batch_size: int, device="cpu", dtype=None) -> dict:
        """
        Initialise KV cache for all layers.
        Returns: dict[str, LayerCache]  (mirrors Gemma's Cache type)
        """
        import torch
        from kv_cache import init_layer_cache
        if dtype is None:
            dtype = torch.bfloat16
        return {
            f"layer_{i}": init_layer_cache(
                cache_size=self.block_size,
                num_kv_heads=self.n_kv_head,
                head_dim=self.head_dim,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            for i in range(self.n_layer)
        }


# ─────────────────────────────────────────────
# Production presets
# ─────────────────────────────────────────────

# 150M params: n_embd=768, n_layer=18, n_head=12, n_kv_head=4
# Expects tiktoken BPE vocab (50,257 tokens)
# block_size=2048 for RAG context injection
MODEL_150M = GPTConfig(
    vocab_size=50_257,          # tiktoken GPT-2 BPE
    n_layer=18,
    n_embd=768,
    n_head=12,
    n_kv_head=4,                # GQA: 3 query heads per kv head
    block_size=2048,            # enough for system prompt + RAG + history
    attention_pattern=GEMMA3_ATTENTION_PATTERN,
    sliding_window_size=256,    # local window for 2048 ctx
    use_post_attn_norm=True,
    use_post_ffw_norm=True,
    use_qk_norm=True,
    attn_logits_soft_cap=50.0,
    final_logit_softcap=None,
    rope_base_frequency=10_000,
    rope_scale_factor=1.0,
    dropout=0.0,
)

# 120M params: fits 12 GB VRAM (batch=4, seq=1024, bf16)
# n_embd=640, n_layer=16, n_head=8, n_kv_head=4
MODEL_120M = GPTConfig(
    vocab_size=50_257,
    n_layer=16,
    n_embd=640,
    n_head=8,
    n_kv_head=4,
    block_size=2048,
    attention_pattern=GEMMA3_ATTENTION_PATTERN,
    sliding_window_size=256,
    use_post_attn_norm=True,
    use_post_ffw_norm=True,
    use_qk_norm=True,
    attn_logits_soft_cap=50.0,
    final_logit_softcap=None,
    rope_base_frequency=10_000,
    rope_scale_factor=1.0,
    dropout=0.0,
)

# ~130M params: n_embd=672, n_layer=18 — fits 12GB VRAM (batch=4, seq=1024, bf16)
MODEL_130M = GPTConfig(
    vocab_size=50_257,
    n_layer=18,
    n_embd=672,
    n_head=8,
    n_kv_head=4,
    block_size=2048,
    attention_pattern=GEMMA3_ATTENTION_PATTERN,
    sliding_window_size=256,
    use_post_attn_norm=True,
    use_post_ffw_norm=True,
    use_qk_norm=True,
    attn_logits_soft_cap=50.0,
    final_logit_softcap=None,
    rope_base_frequency=10_000,
    rope_scale_factor=1.0,
    dropout=0.0,
)

# Tiny debug model (~8M params): fast iteration, no GPU needed
MODEL_TINY = GPTConfig(
    vocab_size=50_257,
    n_layer=4,
    n_embd=256,
    n_head=4,
    n_kv_head=2,
    block_size=512,
    attention_pattern=GEMMA3_ATTENTION_PATTERN,
    sliding_window_size=64,
    use_post_attn_norm=True,
    use_post_ffw_norm=True,
    use_qk_norm=True,
    attn_logits_soft_cap=50.0,
    dropout=0.0,
)
