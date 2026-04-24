"""
KeeKee v3 — Gemma-Inspired Production Chatbot
===================================================
Public API surface.

Core model + production chatbot layers:
  - Chat: multi-turn conversation (ChatSampler, Conversation)
  - RAG: retrieval-augmented generation (RAGRetriever, TextEmbedder)
  - Tools: MCP-style tool calling (Tool, ToolRegistry, GmailTool)
  - Agent: full orchestrator (ChatAgent)
"""

# ── Core model ───────────────────────────────────────────────────────────
from config import GPTConfig, AttentionType, GEMMA3_ATTENTION_PATTERN, MODEL_150M, MODEL_TINY
from layers import RMSNorm, Linear
from rope import RoPECache, apply_rope
from kv_cache import Cache, LayerCache, init_layer_cache, clear_cache
from attention import Attention
from mlp import GeGLU, SwiGLU
from block import Block
from model import KeeKee, ModelOutput
from sampling import SamplingMethod, Greedy, TopKSampling, TopPSampling, TemperatureSampling
from sampler import Sampler, SamplerLoop, SamplingState, prefill
from tokenizer import CharTokenizer, BPETokenizer, SpecialTokens
from router import TaskRouter, GenerationProfile

# ── Chat system ──────────────────────────────────────────────────────────
from chat.conversation import Turn, Conversation
from chat.chat_sampler import ChatSampler

# ── Tools + MCP ──────────────────────────────────────────────────────────
from tools.base import Tool, ToolCall, ToolResult
from tools.registry import ToolRegistry
from mcp.protocol import MCPProtocol

# ── Agent ────────────────────────────────────────────────────────────────
from agent.agent import ChatAgent

__all__ = [
    # Config
    "GPTConfig", "AttentionType", "GEMMA3_ATTENTION_PATTERN",
    "MODEL_150M", "MODEL_TINY",
    # Layers
    "RMSNorm", "Linear",
    # RoPE
    "RoPECache", "apply_rope",
    # KV Cache
    "Cache", "LayerCache", "init_layer_cache", "clear_cache",
    # Attention
    "Attention",
    # FFW
    "GeGLU", "SwiGLU",
    # Block
    "Block",
    # Model
    "KeeKee", "ModelOutput",
    # Sampling
    "SamplingMethod", "Greedy", "TopKSampling", "TopPSampling", "TemperatureSampling",
    # Sampler
    "Sampler", "SamplerLoop", "SamplingState", "prefill",
    # Tokenizer
    "CharTokenizer", "BPETokenizer", "SpecialTokens",
    # Router
    "TaskRouter", "GenerationProfile",
    # Chat
    "Turn", "Conversation", "ChatSampler",
    # Tools
    "Tool", "ToolCall", "ToolResult", "ToolRegistry",
    # MCP
    "MCPProtocol",
    # Agent
    "ChatAgent",
]
