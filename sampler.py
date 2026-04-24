"""
sampler.py — Prefill + Autoregressive Decoding Loop
======================================================
Gemma-inspired (text/_prefill.py + text/_sampler_loop.py + text/_sampler.py):

  SamplingState  — mutable state carried between steps  (mirrors Gemma's SamplingState)
  prefill()      — Phase 1: process full prompt, fill KV cache
  SamplerLoop    — Phase 2: autoregressive token generation loop
  Sampler        — High-level user-facing API (prompt str → response str)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Generator

from config import GPTConfig
from model import KeeKee
from sampling import SamplingMethod, TopKSampling
from kv_cache import Cache, clear_cache, is_cache_full
from tokenizer import CharTokenizer


# ─────────────────────────────────────────────
# SamplingState  (mirrors Gemma SamplingState)
# ─────────────────────────────────────────────
@dataclass
class SamplingState:
    """
    All mutable state for one generation session.
    Passed through the decoding loop step-by-step.

    Mirrors Gemma's SamplingState fields.
    """
    step: int                              # current decoding step
    done: torch.Tensor                     # [B] bool — which sequences finished

    last_token: torch.Tensor               # [B] — last predicted token id
    last_token_pos: torch.Tensor           # [B] — RoPE position of last_token

    predicted_tokens: torch.Tensor         # [B, max_new_tokens] — output buffer
    cache: Cache                           # KV cache (all layers)

    # Prompt info
    prompt_length: int
    max_new_tokens: int

    # Forbidden tokens (e.g. padding id)
    forbidden_token_ids: list[int] = field(default_factory=list)

    @property
    def is_done(self) -> bool:
        """True when ALL batch elements have finished."""
        return bool(self.done.all().item())

    @property
    def is_cache_full(self) -> bool:
        """True when any layer's KV cache is saturated."""
        return is_cache_full(self.cache["layer_0"])


# ─────────────────────────────────────────────
# Phase 1: Prefill
# ─────────────────────────────────────────────
def prefill(
    model: KeeKee,
    tokens: torch.Tensor,       # [B, L] — full prompt token ids
    cache: Cache,
) -> SamplingState:
    """
    Process the entire prompt at once (parallel), fill the KV cache.
    Returns a SamplingState ready for the decoding loop.

    Mirrors Gemma's _prefill.py.
    """
    device = tokens.device
    B, L = tokens.shape

    # Run model on full prompt — return_last_only=True means we only
    # get logits for the LAST token (massive saving vs computing all L positions)
    with torch.no_grad():
        out = model(tokens, cache=cache, return_last_only=True)

    # The last prompt token's logit tells us the FIRST generated token
    # (we don't sample it here — the loop does that)
    last_token = tokens[:, -1]            # [B] — last prompt token
    last_token_pos = torch.full(          # [B] — its position
        (B,), L - 1, dtype=torch.long, device=device
    )

    return SamplingState(
        step=0,
        done=torch.zeros(B, dtype=torch.bool, device=device),
        last_token=last_token,
        last_token_pos=last_token_pos,
        predicted_tokens=torch.zeros(B, 0, dtype=torch.long, device=device),
        cache=out.cache,
        prompt_length=L,
        max_new_tokens=0,   # set by SamplerLoop
    )


# ─────────────────────────────────────────────
# Phase 2: Autoregressive Decoding Loop
# ─────────────────────────────────────────────
class SamplerLoop:
    """
    Autoregressive decoding loop.
    Mirrors Gemma's SamplerLoop / _sample_loop.

    Each step:
        1. Feed only the LAST token (with updated position) to the model
        2. Get logits [B, vocab]
        3. Mask forbidden tokens
        4. Sample next token via SamplingMethod
        5. Update state (done, cache, predicted buffer)
    """

    def __init__(
        self,
        model: KeeKee,
        sampling: SamplingMethod,
        end_token_ids: list[int],
    ):
        self.model = model
        self.sampling = sampling
        self.end_token_ids = end_token_ids

    @torch.no_grad()
    def run(
        self,
        state: SamplingState,
        max_new_tokens: int,
        stream: bool = False,
    ) -> SamplingState | Generator[int, None, SamplingState]:
        """
        Run the generation loop.

        Args:
            state:          SamplingState from prefill()
            max_new_tokens: maximum tokens to generate
            stream:         if True, yield each token as it's produced

        Returns:
            Final SamplingState (or generator if stream=True)
        """
        state.max_new_tokens = max_new_tokens
        state.predicted_tokens = torch.zeros(
            state.last_token.shape[0], max_new_tokens,
            dtype=torch.long, device=state.last_token.device,
        )

        if stream:
            return self._stream_loop(state)
        return self._run_loop(state)

    def _run_loop(self, state: SamplingState) -> SamplingState:
        for _ in range(state.max_new_tokens):
            if state.is_done or state.is_cache_full:
                break
            state = self._sample_step(state)
        return state

    def _stream_loop(self, state: SamplingState) -> Generator:
        for _ in range(state.max_new_tokens):
            if state.is_done or state.is_cache_full:
                break
            state = self._sample_step(state)
            # Yield the last sampled token for each live sequence
            yield state.predicted_tokens[:, state.step - 1]
        return state

    def _sample_step(self, state: SamplingState) -> SamplingState:
        """Single autoregressive step."""
        device = state.last_token.device
        B = state.last_token.shape[0]

        # ── Feed single token to model ─────────────────────────────────
        # Shape [B, 1] — only the last token; cache has the rest
        idx = state.last_token.unsqueeze(1)   # [B, 1]

        out = self.model(idx, cache=state.cache, return_last_only=True)
        logits = out.logits[:, 0, :]   # [B, vocab_size]

        # ── Mask forbidden token ids ────────────────────────────────────
        for tid in state.forbidden_token_ids:
            logits[:, tid] = float("-inf")

        # ── Freeze already-finished sequences ──────────────────────────
        # Their logits are forced to EOS so they don't contribute noise
        if state.done.any():
            eos = self.end_token_ids[0] if self.end_token_ids else 0
            logits[state.done] = float("-inf")
            logits[state.done, eos] = 0.0

        # ── Sample next token ───────────────────────────────────────────
        next_token = self.sampling.get_next_token(logits)   # [B]

        # ── Update predicted buffer ─────────────────────────────────────
        step = state.step
        state.predicted_tokens[:, step] = next_token

        # ── Update done flags ───────────────────────────────────────────
        is_eos = torch.zeros(B, dtype=torch.bool, device=device)
        for eid in self.end_token_ids:
            is_eos |= (next_token == eid)
        state.done = state.done | is_eos

        # ── Advance positions (freeze done sequences) ───────────────────
        state.last_token = next_token
        state.last_token_pos = state.last_token_pos + (~state.done).long()

        state.cache = out.cache
        state.step += 1
        return state


# ─────────────────────────────────────────────
# High-level Sampler  (mirrors Gemma Sampler)
# ─────────────────────────────────────────────
class Sampler:
    """
    User-facing generation API.

    Usage:
        sampler = Sampler(model, tokenizer)
        response = sampler.sample("What is 2+2?", max_new_tokens=64)

    Internally:
        prompt → tokenize → pad → prefill → SamplerLoop → decode → str
    """

    def __init__(
        self,
        model: KeeKee,
        tokenizer: CharTokenizer,
        sampling: Optional[SamplingMethod] = None,
        pad_id: int = 0,
        pad_lengths: tuple[int, ...] = (64, 128, 256, 512),   # bucket sizes
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampling = sampling or TopKSampling(k=50, temperature=0.8)
        self.pad_id = pad_id
        self.pad_lengths = sorted(pad_lengths)

        end_ids = list(tokenizer.special_tokens.end_tokens())
        self.loop = SamplerLoop(model, self.sampling, end_ids)

    # ── Prompt tokenisation + bucket padding ────────────────────────────
    def _tokenize_and_pad(self, prompt: str, device) -> torch.Tensor:
        """
        Encode prompt → token ids, pad to smallest bucket >= len.
        Bucket padding avoids recompilation for every new prompt length.
        (Mirrors Gemma's smart padding in _sampler.py)
        """
        tokens = self.tokenizer.encode(prompt, add_bos=True)
        L = len(tokens)

        # Pick bucket
        bucket = next((b for b in self.pad_lengths if b >= L), self.pad_lengths[-1])
        pad_len = bucket - L
        tokens = tokens + [self.pad_id] * pad_len

        return torch.tensor([tokens], dtype=torch.long, device=device)

    # ── Main sample() call ───────────────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        prompt: str | list[str],
        max_new_tokens: int = 128,
        sampling: Optional[SamplingMethod] = None,
        stream: bool = False,
    ) -> str | list[str]:
        """
        Generate a response to `prompt`.

        Args:
            prompt:         str (single) or list[str] (batch)
            max_new_tokens: max tokens to generate
            sampling:       override the default SamplingMethod
            stream:         if True, returns a generator of token strings

        Returns:
            str or list[str] — the generated continuation
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        # Batch support
        single = isinstance(prompt, str)
        prompts = [prompt] if single else prompt

        # Tokenise + pad each prompt to same length
        batch_tokens = torch.cat([
            self._tokenize_and_pad(p, device) for p in prompts
        ], dim=0)   # [B, padded_L]

        # Initialise KV cache
        B = batch_tokens.shape[0]
        cache = self.model.init_cache(B)

        # Phase 1: Prefill
        state = prefill(self.model, batch_tokens, cache)

        # Override sampling if provided
        loop = self.loop
        if sampling is not None:
            end_ids = list(self.tokenizer.special_tokens.end_tokens())
            loop = SamplerLoop(self.model, sampling, end_ids)

        # Phase 2: Decode
        state = loop.run(state, max_new_tokens=max_new_tokens)

        # Decode output tokens
        results = []
        for b in range(B):
            toks = state.predicted_tokens[b].tolist()
            # Trim at EOS
            end_ids_set = set(self.tokenizer.special_tokens.end_tokens())
            trimmed = []
            for t in toks:
                if t in end_ids_set or t == 0:
                    break
                trimmed.append(t)
            results.append(self.tokenizer.decode(trimmed))

        return results[0] if single else results
