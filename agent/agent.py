"""
agent.py — ChatAgent: Full Agent Orchestrator
================================================
Gemma-inspired (text/_tool_sampler.py ToolSampler):
  - Extends ChatSampler with RAG retrieval + MCP tool dispatch
  - Agent loop: generate → check for tool calls → execute → re-generate
  - Configurable max_tool_rounds to prevent infinite loops
  - Interactive REPL for terminal chat

Architecture:
    user message
        ↓ RAG retriever (if docs loaded)
        ↓ build prompt (system + context + history + user)
        ↓ ChatSampler.chat() → model.generate()
        ↓ MCPProtocol.parse_tool_calls()
        ├── if tool call → execute → inject result → re-generate (loop)
        └── if plain text → return to user
"""

from __future__ import annotations

import sys
from typing import Optional

from model import KeeKee
from sampling import SamplingMethod, TopKSampling
from chat.chat_sampler import ChatSampler
from chat.conversation import Conversation
from mcp.protocol import MCPProtocol
from tools.base import Tool, ToolResult
from tools.registry import ToolRegistry


# ─────────────────────────────────────────────
# Default system prompt
# ─────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = """\
You are KeeKee, a helpful AI assistant. You answer questions clearly and concisely.

When you need to perform actions (like reading email), use the available tools \
by outputting a tool call in the format:
<tool>{"name": "tool_name", "args": {"param": "value"}}</tool>

After receiving a tool result, use the information to formulate your response.
Do not make up information — rely on tool results and provided context."""


# ─────────────────────────────────────────────
# ChatAgent
# ─────────────────────────────────────────────
class ChatAgent:
    """
    Full agent: chat + RAG + tool calling.

    Mirrors Gemma's ToolSampler pattern — extends chat sampling with
    tool dispatch loop and RAG context injection.

    The agent loop:
        1. (Optional) Retrieve relevant documents via RAG
        2. Generate a response via ChatSampler
        3. Check for tool calls in the response
        4. If tool calls found → execute → inject result → regenerate (up to max_tool_rounds)
        5. Return final response to user

    Usage:
        agent = ChatAgent(model, tokenizer)
        agent.add_documents(["Python is a programming language."])
        response = agent.chat("What is Python?")

        # With tools:
        from tools.mail import GmailTool
        agent = ChatAgent(model, tokenizer, tools=[GmailTool()])
        response = agent.chat("Read my latest emails")

        # Interactive REPL:
        agent.interactive()
    """

    def __init__(
        self,
        model: KeeKee,
        tokenizer,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: Optional[list[Tool]] = None,
        retriever=None,  # RAGRetriever (optional import)
        max_tool_rounds: int = 3,
        sampling: Optional[SamplingMethod] = None,
        max_history_tokens: int = 1536,
    ):
        """
        Args:
            model:              KeeKee model instance
            tokenizer:          BPETokenizer or CharTokenizer
            system_prompt:      system prompt text
            tools:              list of Tool instances to register
            retriever:          RAGRetriever instance (or None)
            max_tool_rounds:    max tool call → re-generate iterations
            sampling:           SamplingMethod for generation
            max_history_tokens: max tokens for conversation history
        """
        # Tool registry + MCP protocol
        self.registry = ToolRegistry()
        self.mcp = MCPProtocol(self.registry)
        self.max_tool_rounds = max_tool_rounds

        # Register tools
        for tool in (tools or []):
            self.registry.register(tool)

        # Build full system prompt (base + tool descriptions)
        full_system_prompt = self._build_system_prompt(system_prompt)

        # Chat sampler (handles conversation state + generation)
        self.chat_sampler = ChatSampler(
            model=model,
            tokenizer=tokenizer,
            sampling=sampling,
            multi_turn=True,
            system_prompt=full_system_prompt,
            max_history_tokens=max_history_tokens,
        )

        # RAG retriever (optional)
        self.retriever = retriever

    def _build_system_prompt(self, base_prompt: str) -> str:
        """
        Build the full system prompt by appending tool descriptions.

        Args:
            base_prompt: base system prompt text

        Returns:
            Complete system prompt with tool descriptions appended
        """
        parts = [base_prompt]

        tool_section = self.registry.system_prompt_section()
        if tool_section:
            parts.append("\n" + tool_section)

        return "\n".join(parts)

    def chat(
        self,
        message: str,
        max_new_tokens: int = 256,
        sampling: Optional[SamplingMethod] = None,
    ) -> str:
        """
        Send a message and get a response.

        Handles RAG retrieval and tool call loop automatically.

        Args:
            message:        user message text
            max_new_tokens: max tokens to generate per round
            sampling:       override default sampling method

        Returns:
            Final model response text (after any tool call rounds)
        """
        # ── 1. RAG: retrieve context if retriever is loaded ──────────────
        context = ""
        if self.retriever is not None:
            try:
                if self.retriever.store.size > 0:
                    context = self.retriever.retrieve(message)
            except Exception:
                pass  # RAG failure shouldn't block chat

        # ── 2. Generate initial response ─────────────────────────────────
        response = self.chat_sampler.chat(
            prompt=message,
            context=context,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
        )

        # ── 3. Tool call loop ────────────────────────────────────────────
        for round_num in range(self.max_tool_rounds):
            if not self.mcp.has_tool_call(response):
                break

            # Parse and execute tool calls
            calls = self.mcp.parse_tool_calls(response)
            if not calls:
                break

            results = self.mcp.execute_calls(calls)
            result_text = self.mcp.format_results(results)

            # Get the non-tool part of the response (model's reasoning)
            reasoning = self.mcp.extract_non_tool_text(response)

            # Re-generate with tool results injected
            # The tool result is sent as a "tool" turn in the conversation
            tool_msg = result_text
            if reasoning:
                tool_msg = f"{reasoning}\n{result_text}"

            response = self.chat_sampler.chat(
                prompt=tool_msg,
                context="",
                max_new_tokens=max_new_tokens,
                sampling=sampling,
            )

        return response

    def add_documents(
        self,
        texts: list[str],
        metadata: Optional[list[dict]] = None,
    ) -> int:
        """
        Add documents to the RAG store.

        If no retriever was provided at init, creates one automatically.

        Args:
            texts:    list of document strings
            metadata: optional per-document metadata

        Returns:
            Number of chunks indexed
        """
        if self.retriever is None:
            # Lazy import to avoid requiring sentence-transformers if not used
            from rag.retriever import RAGRetriever
            self.retriever = RAGRetriever()

        return self.retriever.add_documents(texts, metadata)

    def reset(self) -> None:
        """Clear conversation history (keeps tools and RAG docs)."""
        self.chat_sampler.reset()

    def reset_all(self) -> None:
        """Clear everything: conversation, RAG store."""
        self.chat_sampler.reset()
        if self.retriever is not None:
            self.retriever.clear()

    def interactive(self) -> None:
        """
        Start an interactive REPL chat loop in the terminal.

        Commands:
            /quit, /exit  — exit the loop
            /reset        — clear conversation history
            /clear        — clear everything (history + RAG)
            /docs <text>  — add inline document to RAG
            /help         — show available commands
        """
        print("╔══════════════════════════════════════════════╗")
        print("║           KeeKee Chat Agent v3              ║")
        print("║  Type /help for commands, /quit to exit      ║")
        print("╚══════════════════════════════════════════════╝")
        print()

        if self.registry.size > 0:
            print(f"🔧 Tools: {', '.join(self.registry.tool_names)}")
        if self.retriever and self.retriever.store.size > 0:
            print(f"📚 RAG: {self.retriever.chunk_count} chunks indexed")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]

                if cmd in ("/quit", "/exit"):
                    print("Goodbye!")
                    break
                elif cmd == "/reset":
                    self.reset()
                    print("[Conversation history cleared]")
                    continue
                elif cmd == "/clear":
                    self.reset_all()
                    print("[All state cleared]")
                    continue
                elif cmd == "/docs":
                    doc_text = user_input[5:].strip()
                    if doc_text:
                        n = self.add_documents([doc_text])
                        print(f"[Added {n} chunks to RAG store]")
                    else:
                        print("[Usage: /docs <document text>]")
                    continue
                elif cmd == "/help":
                    print("Commands:")
                    print("  /quit, /exit  — exit")
                    print("  /reset        — clear conversation history")
                    print("  /clear        — clear everything")
                    print("  /docs <text>  — add document to RAG")
                    print("  /help         — show this help")
                    continue
                else:
                    print(f"[Unknown command: {cmd}. Type /help for help.]")
                    continue

            # Generate response
            try:
                response = self.chat(user_input)
                print(f"\nKeeKee: {response}\n")
            except Exception as e:
                print(f"\n[Error: {type(e).__name__}: {e}]\n")

    def __repr__(self) -> str:
        tools = self.registry.tool_names
        rag = self.retriever.chunk_count if self.retriever else 0
        turns = self.chat_sampler.conversation.turn_count
        return (
            f"ChatAgent(turns={turns}, tools={tools}, rag_chunks={rag})"
        )
