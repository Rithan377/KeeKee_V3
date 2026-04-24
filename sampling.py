"""
sampling.py — Sampling Strategies
====================================
Gemma-inspired (text/_sampling.py):
  - Abstract base class SamplingMethod
  - Greedy: argmax
  - TopKSampling: keep top-k logits, sample
  - TopPSampling: nucleus — keep smallest set whose cumprob >= p
  - TemperatureSampling: full vocab with temperature rescaling
Each strategy is a standalone class — swap them freely in Sampler.
"""

import abc
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Abstract base class  (mirrors Gemma SamplingMethod)
# ─────────────────────────────────────────────
class SamplingMethod(abc.ABC):
    """
    Base class for token sampling strategies.
    All subclasses implement `get_next_token(logits)`.
    """

    @abc.abstractmethod
    def get_next_token(
        self,
        logits: torch.Tensor,               # [B, vocab_size]
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:                       # [B]
        """Sample the next token id for each batch element."""
        ...


# ─────────────────────────────────────────────
# Greedy  (mirrors Gemma Greedy)
# ─────────────────────────────────────────────
class Greedy(SamplingMethod):
    """Always pick the highest-probability token. Zero randomness."""

    def get_next_token(self, logits: torch.Tensor, rng=None) -> torch.Tensor:
        return logits.argmax(dim=-1)   # [B]


# ─────────────────────────────────────────────
# Temperature Sampling
# ─────────────────────────────────────────────
@dataclass
class TemperatureSampling(SamplingMethod):
    """Sample from full vocab after scaling by temperature."""
    temperature: float = 1.0

    def get_next_token(self, logits: torch.Tensor, rng=None) -> torch.Tensor:
        scaled = logits / max(self.temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)


# ─────────────────────────────────────────────
# Top-K Sampling  (mirrors Gemma TopkSampling)
# ─────────────────────────────────────────────
@dataclass
class TopKSampling(SamplingMethod):
    """
    Keep the top-k logits, zero out the rest, then sample.
    Equivalent to Gemma's TopkSampling(k=..., temperature=...).
    """
    k: int = 50
    temperature: float = 1.0

    def get_next_token(self, logits: torch.Tensor, rng=None) -> torch.Tensor:
        k = min(self.k, logits.size(-1))

        # Get the k-th largest logit value as threshold
        topk_vals, _ = torch.topk(logits, k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)   # [B, 1]

        # Mask everything below threshold to -inf
        logits = logits.masked_fill(logits < threshold, float("-inf"))

        # Temperature + sample
        scaled = logits / max(self.temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)


# ─────────────────────────────────────────────
# Top-P (Nucleus) Sampling  (mirrors Gemma TopPSampling)
# ─────────────────────────────────────────────
@dataclass
class TopPSampling(SamplingMethod):
    """
    Nucleus sampling: keep smallest set of tokens whose cumulative
    probability >= p, then sample from them.

    Equivalent to Gemma's TopPSampling(p=..., temperature=...).
    """
    p: float = 0.9
    temperature: float = 1.0

    def get_next_token(self, logits: torch.Tensor, rng=None) -> torch.Tensor:
        # Temperature scaling
        scaled = logits / max(self.temperature, 1e-8)

        # Sort descending
        probs = F.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)

        # Cumulative probability
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # Mask: keep tokens BEFORE cumprob exceeds p
        # (shift right by 1 so we always keep the first token)
        remove_mask = (cum_probs - sorted_probs) >= self.p
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

        # Normalise the nucleus
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Sample from sorted, then un-sort
        sampled = torch.multinomial(sorted_probs, num_samples=1, generator=rng)  # [B, 1]
        next_token = sorted_idx.gather(-1, sampled).squeeze(-1)                  # [B]
        return next_token


# ─────────────────────────────────────────────
# Convenience factory  (maps router profile → SamplingMethod)
# ─────────────────────────────────────────────
def sampling_from_profile(profile: dict) -> SamplingMethod:
    """
    Build a SamplingMethod from a router profile dict.
    Used by Sampler to pick strategy automatically.
    """
    temperature = profile.get("temperature", 1.0)
    top_k = profile.get("top_k", None)
    top_p = profile.get("top_p", None)

    if temperature == 0.0:
        return Greedy()
    if top_p is not None:
        return TopPSampling(p=top_p, temperature=temperature)
    if top_k is not None:
        return TopKSampling(k=top_k, temperature=temperature)
    return TemperatureSampling(temperature=temperature)
