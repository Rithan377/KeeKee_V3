"""
chat_sampler.py — Stateful Multi-Turn Chat Sampler
=====================================================
Gemma-inspired (text/_chat_sampler.py ChatSampler):
  - Stateful: tracks conversation across .chat() calls
  - Auto-formats prompts with turn tags
  - Manages KV cache lifecycle
  - Supports multi-turn and single-turn modes

Unlike Gemma's JAX-based cache reuse across turns, this PyTorch
implementation re-encodes the trimmed history each turn (simpler,
no cache state corruption risk). Conversation history provides continuity.
"""

from __future__ import annotations

import sys
from typing import Optional, Union

import torch

from config import GPTConfig
from model import KeeKee
from sampling import SamplingMethod, TopKSampling
from sampler import Sampler, SamplerLoop, prefill
from chat.conversation import Conversation


# ─────────────────────────────────────────────
# ChatSampler  (mirrors Gemma ChatSampler)
# ─────────────────────────────────────────────
class ChatSampler:
    """
    Stateful multi-turn chat sampler.

    Wraps the existing Sampler with conversation state management.
    Each .chat() call:
      1. Builds the full prompt from conversation history
      2. Tokenizes → prefill → decode
      3. Updates conversation with user + model turns
      4. Returns the model's response text

    Usage:
        sampler = ChatSampler(model, tokenizer)
        r1 = sampler.chat("Hello!")
        r2 = sampler.chat("Tell me more.")  # remembers previous turn
        sampler.reset()  # clear history

    Mirrors Gemma's ChatSampler pattern but simplified for PyTorch.
    """

    def __init__(
        self,
        model: KeeKee,
        tokenizer,  # CharTokenizer or BPETokenizer
        sampling: Optional[SamplingMethod] = None,
        multi_turn: bool = True,
        system_prompt: str = "",
        max_history_tokens: int = 1536,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sampling = sampling or TopKSampling(k=50, temperature=0.8)
        self.multi_turn = multi_turn

        # Conversation state
        self.conversation = Conversation(
            system_prompt=system_prompt,
            max_history_tokens=max_history_tokens,
        )

        # Compute generation budget from config
        self._block_size = model.config.block_size

    def chat(
        self,
        prompt: str,
        context: str = "",
        max_new_tokens: int = 256,
        sampling: Optional[SamplingMethod] = None,
        stream: bool = False,
    ) -> str:
        """
        Send a message and get a response.

        Args:
            prompt:         user message text
            context:        RAG context to inject (pre-formatted)
            max_new_tokens: max tokens to generate
            sampling:       override default sampling method
            stream:         if True, print tokens as they're generated

        Returns:
            Model's response text
        """
        device = next(self.model.parameters()).device
        self.model.eval()

        # ── 1. Build full prompt ─────────────────────────────────────────
        if not self.multi_turn:
            self.conversation.clear()

        full_prompt = self.conversation.format_prompt(
            self.tokenizer, prompt, context=context
        )

        # ── 2. Tokenize ─────────────────────────────────────────────────
        tokens = self.tokenizer.encode(full_prompt, add_bos=True)

        # Ensure we don't exceed block_size
        if len(tokens) + max_new_tokens > self._block_size:
            # Trim tokens from the beginning (keep most recent context)
            max_prompt_len = self._block_size - max_new_tokens
            if max_prompt_len < 1:
                max_prompt_len = 1
            tokens = tokens[-max_prompt_len:]

        # Pad to a reasonable bucket size for efficient computation
        token_tensor = torch.tensor(
            [tokens], dtype=torch.long, device=device
        )

        # ── 3. Init cache + prefill ──────────────────────────────────────
        cache = self.model.init_cache(batch_size=1)

        with torch.no_grad():
            state = prefill(self.model, token_tensor, cache)

        # ── 4. Decode ────────────────────────────────────────────────────
        end_ids = list(self.tokenizer.special_tokens.end_tokens())
        active_sampling = sampling or self.sampling
        loop = SamplerLoop(self.model, active_sampling, end_ids)

        if stream:
            state = loop.run(state, max_new_tokens=max_new_tokens, stream=False)
        else:
            state = loop.run(state, max_new_tokens=max_new_tokens)

        # ── 5. Decode output tokens ──────────────────────────────────────
        output_ids = state.predicted_tokens[0].tolist()

        # Trim at EOS / end-of-turn / pad
        end_ids_set = set(end_ids) | {0}
        trimmed = []
        for t in output_ids:
            if t in end_ids_set:
                break
            trimmed.append(t)

        response = self.tokenizer.decode(trimmed)

        # ── 6. Update conversation ───────────────────────────────────────
        self.conversation.add_turn("user", prompt)
        self.conversation.add_turn("model", response)

        if stream:
            print(response, flush=True)

        return response

    def reset(self) -> None:
        """Clear conversation history."""
        self.conversation.clear()

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt (also clears history)."""
        self.conversation.system_prompt = prompt
        self.conversation.clear()

    @property
    def history(self) -> list:
        """Access raw conversation turns."""
        return self.conversation.turns

    def __repr__(self) -> str:
        return (
            f"ChatSampler(multi_turn={self.multi_turn}, "
            f"turns={self.conversation.turn_count}, "
            f"sampling={self.sampling.__class__.__name__})"
        )
