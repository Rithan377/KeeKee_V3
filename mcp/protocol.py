"""
protocol.py — MCP-Style Tool Protocol
========================================
Gemma-inspired (tools/_manager.py maybe_execute_tool):
  - Parse <tool>{"name": ..., "args": {...}}</tool> from model output
  - Dispatch parsed calls through ToolRegistry
  - Format results as <tool_result>...</tool_result> for re-injection

This implements a simplified MCP (Model Context Protocol) that works
with text-based tool calling — the model generates structured JSON
within <tool> tags, and results are injected back as <tool_result> blocks.

Protocol format:
    Model output:  <tool>{"name": "gmail", "args": {"action": "read_inbox"}}</tool>
    Injected back: <tool_result>{"name": "gmail", "success": true, "output": "..."}</tool_result>
"""

from __future__ import annotations

import json
import re
from typing import Optional

from tools.base import ToolCall, ToolResult
from tools.registry import ToolRegistry


# ─────────────────────────────────────────────
# MCPProtocol
# ─────────────────────────────────────────────
class MCPProtocol:
    """
    MCP-style tool protocol: parse calls from model output, dispatch,
    and format results for re-injection into the conversation.

    Mirrors Gemma's _manager.py maybe_execute_tool() pattern but
    uses simple XML-like tags instead of the full dialog library.

    Protocol:
        1. Model generates: <tool>{"name": "...", "args": {...}}</tool>
        2. MCPProtocol.parse_tool_calls() extracts the JSON
        3. MCPProtocol.execute_calls() dispatches to ToolRegistry
        4. MCPProtocol.format_results() wraps results in <tool_result> tags
        5. Agent injects the result text back into the conversation

    Usage:
        protocol = MCPProtocol(registry)

        if protocol.has_tool_call(model_output):
            calls = protocol.parse_tool_calls(model_output)
            results = protocol.execute_calls(calls)
            result_text = protocol.format_results(results)
            # → inject result_text into next prompt
    """

    TOOL_TAG_OPEN = "<tool>"
    TOOL_TAG_CLOSE = "</tool>"
    RESULT_TAG_OPEN = "<tool_result>"
    RESULT_TAG_CLOSE = "</tool_result>"

    # Regex to extract tool call JSON from model output
    _TOOL_PATTERN = re.compile(
        r"<tool>\s*(.*?)\s*</tool>",
        re.DOTALL,
    )

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def has_tool_call(self, text: str) -> bool:
        """
        Quick check if text contains a tool call.

        Args:
            text: model output text

        Returns:
            True if at least one <tool>...</tool> block is found
        """
        return self.TOOL_TAG_OPEN in text and self.TOOL_TAG_CLOSE in text

    def parse_tool_calls(self, text: str) -> list[ToolCall]:
        """
        Extract tool calls from model output text.

        Finds all <tool>{"name": ..., "args": {...}}</tool> blocks
        and parses them into ToolCall objects.

        Args:
            text: model output text

        Returns:
            list of ToolCall objects (empty if no valid calls found)
        """
        calls = []

        for match in self._TOOL_PATTERN.finditer(text):
            raw_json = match.group(1).strip()

            try:
                data = json.loads(raw_json)
            except json.JSONDecodeError:
                # Try to fix common issues: single quotes, trailing commas
                try:
                    fixed = raw_json.replace("'", '"')
                    # Remove trailing commas before } or ]
                    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
                    data = json.loads(fixed)
                except json.JSONDecodeError:
                    continue  # Skip unparseable calls

            # Validate structure
            if not isinstance(data, dict) or "name" not in data:
                continue

            calls.append(ToolCall(
                name=str(data["name"]),
                arguments=data.get("args", data.get("arguments", {})),
            ))

        return calls

    def execute_calls(self, calls: list[ToolCall]) -> list[ToolResult]:
        """
        Dispatch each tool call through the registry.

        Args:
            calls: list of parsed ToolCall objects

        Returns:
            list of ToolResult objects (one per call, in order)
        """
        results = []

        for call in calls:
            result = self.registry.dispatch(call)
            results.append(result)

        return results

    def format_results(self, results: list[ToolResult]) -> str:
        """
        Format tool results for re-injection into the conversation.

        Each result is wrapped in <tool_result>...</tool_result> tags
        as a structured JSON block that the model can parse.

        Args:
            results: list of ToolResult objects

        Returns:
            Formatted string with all results
        """
        parts = []

        for result in results:
            result_data = {
                "name": result.name,
                "success": result.success,
                "output": result.output,
            }
            parts.append(
                f"{self.RESULT_TAG_OPEN}"
                f"{json.dumps(result_data)}"
                f"{self.RESULT_TAG_CLOSE}"
            )

        return "\n".join(parts)

    def extract_non_tool_text(self, text: str) -> str:
        """
        Extract the non-tool-call portion of model output.

        Useful for getting the model's natural language response
        that may surround tool calls.

        Args:
            text: model output text

        Returns:
            Text with <tool>...</tool> blocks removed
        """
        cleaned = self._TOOL_PATTERN.sub("", text)
        return cleaned.strip()

    def __repr__(self) -> str:
        return f"MCPProtocol(tools={self.registry.tool_names})"
