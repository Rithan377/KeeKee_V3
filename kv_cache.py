"""
kv_cache.py — KV Cache Management
====================================
Gemma-inspired (utils/_cache_helper.py + nn/_modules.py):
  - LayerCache: dict with 'k', 'v', 'end_index' keys (same as Gemma)
  - Cache: dict[str, LayerCache] — one entry per transformer layer
  - Dynamic update using torch equivalent of jax.lax.dynamic_update_slice
  - is_full check, end_index tracking
"""

import torch
from typing import Optional

# ─────────────────────────────────────────────
# Type aliases  (mirrors Gemma's LayerCache / Cache)
# ─────────────────────────────────────────────
# LayerCache keys:
#   'k'         : [batch, cache_size, n_kv_head, head_dim]
#   'v'         : [batch, cache_size, n_kv_head, head_dim]
#   'end_index' : [batch]  — how many tokens written so far
LayerCache = dict[str, torch.Tensor]
Cache = dict[str, LayerCache]          # keyed by "layer_0", "layer_1", ...


# ─────────────────────────────────────────────
# Initialise a single layer's cache
# ─────────────────────────────────────────────
def init_layer_cache(
    cache_size: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> LayerCache:
    """
    Allocate zeroed KV cache for one transformer layer.
    Mirrors Gemma's Attention.init_cache().
    """
    return {
        "k": torch.zeros(batch_size, cache_size, num_kv_heads, head_dim,
                         dtype=dtype, device=device),
        "v": torch.zeros(batch_size, cache_size, num_kv_heads, head_dim,
                         dtype=dtype, device=device),
        "end_index": torch.zeros(batch_size, dtype=torch.int32, device=device),
    }


# ─────────────────────────────────────────────
# Cache update (write new K,V tokens into cache)
# Mirrors Gemma's jax.lax.dynamic_update_slice logic
# ─────────────────────────────────────────────
def update_cache(
    cache: LayerCache,
    new_k: torch.Tensor,   # [B, T, n_kv_head, head_dim]
    new_v: torch.Tensor,   # [B, T, n_kv_head, head_dim]
) -> tuple[torch.Tensor, torch.Tensor, LayerCache]:
    """
    Write new_k, new_v into the KV cache at the current end_index.
    Returns the full (k, v) from cache after update, and the updated cache.
    """
    B, T, H, D = new_k.shape
    cache_size = cache["k"].shape[1]

    end_idx = cache["end_index"][0].item()   # scalar (same for whole batch)
    write_idx = int(end_idx) % cache_size    # wrap-around for rolling cache

    # Write slice: cache[write_idx : write_idx+T]
    cache["k"][:, write_idx: write_idx + T] = new_k
    cache["v"][:, write_idx: write_idx + T] = new_v
    cache["end_index"] = cache["end_index"] + T

    # Return full K, V for attention computation
    return cache["k"], cache["v"], cache


# ─────────────────────────────────────────────
# Cache helper methods  (mirrors Gemma Cache wrapper)
# ─────────────────────────────────────────────
def get_cache_end_index(cache: LayerCache) -> int:
    """Returns how many tokens are currently stored in this layer's cache."""
    return int(cache["end_index"][0].item())


def is_cache_full(cache: LayerCache) -> bool:
    """Returns True if the cache is at capacity."""
    return get_cache_end_index(cache) >= cache["k"].shape[1] - 1


def clear_layer_cache(cache: LayerCache) -> None:
    """Zero out a single layer's cache in-place."""
    cache["k"].zero_()
    cache["v"].zero_()
    cache["end_index"].zero_()


def clear_cache(full_cache: Cache) -> None:
    """Zero out all layers in the cache in-place."""
    for layer_cache in full_cache.values():
        clear_layer_cache(layer_cache)
