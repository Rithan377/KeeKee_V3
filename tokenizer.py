"""
tokenizer.py — BPE Tokenizer (tiktoken-based)
==============================================
Gemma-inspired interface but using Byte Pair Encoding (BPE) for real-world efficiency.
Replaces the old CharTokenizer.
"""

from __future__ import annotations
import torch
import tiktoken
from dataclasses import dataclass, field
from typing import Optional, List, Set


@dataclass(frozen=True)
class SpecialTokens:
    """
    Special token strings and their eventual IDs.
    We use GPT-2 base + custom turn tokens.
    """
    # Names of special tokens
    pad: str = "<|pad|>"
    bos: str = "<|bos|>"
    eos: str = "<|eos|>"
    unk: str = "<|unk|>"

    # Conversation turn tokens (Gemma-style)
    start_user: str = "<start_of_turn>user"
    start_model: str = "<start_of_turn>model"
    end_turn: str = "<end_of_turn>"

    def to_dict(self) -> dict:
        return {
            self.pad: 50257,
            self.bos: 50258,
            self.eos: 50259,
            self.unk: 50260,
            self.start_user: 50261,
            self.start_model: 50262,
            self.end_turn: 50263,
        }


class BPETokenizer:
    """
    BPE Tokenizer wrapping tiktoken (GPT-2 encoding).
    Supports special tokens for chat turns and padding.
    """

    def __init__(self, model_name: str = "gpt2"):
        # Load base GPT-2 encoding
        self.base_enc = tiktoken.get_encoding(model_name)
        self.special_tokens = SpecialTokens()
        
        # Register special tokens
        self.encoder = tiktoken.Encoding(
            name="microgpt_v3",
            pat_str=self.base_enc._pat_str,
            mergeable_ranks=self.base_enc._mergeable_ranks,
            special_tokens={
                **self.base_enc._special_tokens,
                **self.special_tokens.to_dict()
            }
        )
        
        # ID shortcuts
        spec_dict = self.special_tokens.to_dict()
        self.pad_id = spec_dict[self.special_tokens.pad]
        self.bos_id = spec_dict[self.special_tokens.bos]
        self.eos_id = spec_dict[self.special_tokens.eos]
        self.unk_id = spec_dict[self.special_tokens.unk]
        self.user_id = spec_dict[self.special_tokens.start_user]
        self.model_id = spec_dict[self.special_tokens.start_model]
        self.end_turn_id = spec_dict[self.special_tokens.end_turn]

    @property
    def vocab_size(self) -> int:
        return self.encoder.n_vocab

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allowed_special: Set[str] = "all"
    ) -> List[int]:
        """Convert text to BPE token IDs."""
        ids = self.encoder.encode(text, allowed_special=allowed_special)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Convert token IDs back to text."""
        # Convert to list if it's a tensor
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
            
        if skip_special:
            # Filter out all special token IDs
            special_ids = set(self.special_tokens.to_dict().values())
            # Also include tiktoken's default special tokens
            special_ids.update(self.base_enc._special_tokens.values())
            ids = [i for i in ids if i not in special_ids]
            
        return self.encoder.decode(ids)

    def encode_chat_turn(self, role: str, content: str, add_bos: bool = False) -> List[int]:
        """
        Gemma-style turn encoding:
        <start_of_turn>role\ncontent<end_of_turn>
        """
        role_id = self.user_id if role == "user" else self.model_id
        
        ids = [role_id]
        ids.extend(self.encoder.encode("\n" + content, allowed_special="all"))
        ids.append(self.end_turn_id)
        
        if add_bos:
            ids = [self.bos_id] + ids
        return ids

    def end_tokens(self) -> Set[int]:
        """Tokens that signal the end of a response."""
        return {self.eos_id, self.end_turn_id}
