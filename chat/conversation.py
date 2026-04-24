"""
conversation.py — Multi-Turn Conversation State
==================================================
Gemma-inspired (text/_template.py Turn/Prompt/Response + _chat_sampler.py turns):
  - Turn dataclass: role + content + timestamp
  - Conversation: stateful turn management, prompt formatting, history trimming

Conversation format (Gemma-style):
    <bos><start_of_turn>user\n{msg}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>\n...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Union


# ─────────────────────────────────────────────
# Turn  (mirrors Gemma _template.py Turn)
# ─────────────────────────────────────────────
@dataclass
class Turn:
    """
    A single conversation turn.

    Attributes:
        role:      "user", "model", "system", or "tool"
        content:   the text content of this turn
        timestamp: when this turn was created (auto-set)
    """
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def format(self) -> str:
        """
        Format this turn as a Gemma-style tagged string.

        System turns are formatted differently (no start/end tags, just raw text).
        Tool results are formatted as tool-role turns.
        """
        if self.role == "system":
            return self.content + "\n"
        # <start_of_turn>{role}\n{content}<end_of_turn>\n
        return f"<start_of_turn>{self.role}\n{self.content}<end_of_turn>\n"

    def token_count_estimate(self) -> int:
        """
        Rough token count estimate (~4 chars per BPE token on average).
        Used for budget-based trimming without needing the tokenizer.
        """
        return len(self.format()) // 4 + 1


# ─────────────────────────────────────────────
# Conversation  (mirrors Gemma ChatSampler.turns)
# ─────────────────────────────────────────────
class Conversation:
    """
    Multi-turn conversation state manager.

    Manages the list of turns, formats prompts with system prompt + RAG context
    + conversation history, and trims history to stay within a token budget.

    Usage:
        conv = Conversation(system_prompt="You are a helpful assistant.")
        conv.add_turn("user", "Hello!")
        conv.add_turn("model", "Hi there!")
        prompt = conv.format_prompt(tokenizer, "How are you?")
    """

    def __init__(
        self,
        system_prompt: str = "",
        max_history_tokens: int = 1536,
    ):
        self.turns: list[Turn] = []
        self.system_prompt = system_prompt
        self.max_history_tokens = max_history_tokens

    def add_turn(self, role: str, content: str) -> None:
        """Add a new turn to the conversation."""
        self.turns.append(Turn(role=role, content=content))

    def format_prompt(
        self,
        tokenizer,
        new_user_msg: str,
        context: str = "",
    ) -> str:
        """
        Build the complete prompt string for the model.

        Structure:
            [system prompt]
            [CONTEXT]...retrieved docs...[/CONTEXT]   (if context provided)
            <start_of_turn>user\n{history_msg_1}<end_of_turn>
            <start_of_turn>model\n{history_resp_1}<end_of_turn>
            ...
            <start_of_turn>user\n{new_user_msg}<end_of_turn>
            <start_of_turn>model\n

        Args:
            tokenizer:    tokenizer with encode() for accurate token counting
            new_user_msg: the new user message to append
            context:      RAG context string (already formatted)

        Returns:
            Complete prompt string ready for tokenization
        """
        parts = []

        # 1. System prompt
        if self.system_prompt:
            parts.append(self.system_prompt + "\n")

        # 2. RAG context injection
        if context:
            parts.append(f"[CONTEXT]\n{context}\n[/CONTEXT]\n")

        # 3. Conversation history (trimmed to budget)
        history_turns = self._trim_history(tokenizer)
        for turn in history_turns:
            parts.append(turn.format())

        # 4. New user message
        parts.append(f"<start_of_turn>user\n{new_user_msg}<end_of_turn>\n")

        # 5. Model turn prefix (model starts generating after this)
        parts.append("<start_of_turn>model\n")

        return "".join(parts)

    def _trim_history(self, tokenizer) -> list[Turn]:
        """
        Trim conversation history to fit within max_history_tokens.

        Strategy: drop oldest turns first (sliding window), always keeping
        the most recent turns. This mirrors how real chat systems handle
        context window limits.

        Uses actual tokenizer for accurate counting when available,
        falls back to character-based estimate.
        """
        if not self.turns:
            return []

        # Calculate token budget for history
        budget = self.max_history_tokens

        # Try to fit as many recent turns as possible
        selected: list[Turn] = []
        used_tokens = 0

        for turn in reversed(self.turns):
            # Get accurate token count
            turn_text = turn.format()
            try:
                turn_tokens = len(tokenizer.encode(turn_text))
            except Exception:
                turn_tokens = turn.token_count_estimate()

            if used_tokens + turn_tokens > budget:
                break
            selected.append(turn)
            used_tokens += turn_tokens

        # Reverse back to chronological order
        selected.reverse()
        return selected

    def get_last_response(self) -> Optional[str]:
        """Get the last model response, if any."""
        for turn in reversed(self.turns):
            if turn.role == "model":
                return turn.content
        return None

    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns.clear()

    @property
    def turn_count(self) -> int:
        """Number of turns in the conversation."""
        return len(self.turns)

    def __repr__(self) -> str:
        return (
            f"Conversation(turns={self.turn_count}, "
            f"system_prompt={'set' if self.system_prompt else 'none'})"
        )
