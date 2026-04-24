"""
base.py — Tool ABC and Data Classes
======================================
Gemma-inspired (tools/_manager.py ToolHandlerBase):
  - Tool: abstract base class for all tools
  - ToolCall: parsed tool invocation (name + args)
  - ToolResult: tool execution result (name + success + output)

Every tool must define:
  - name: unique identifier
  - description: human-readable (goes into system prompt)
  - parameters_schema: JSON Schema dict for arguments
  - execute(**kwargs) → ToolResult
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────
# ToolCall  (parsed from model output)
# ─────────────────────────────────────────────
@dataclass
class ToolCall:
    """
    A parsed tool invocation extracted from model output.

    Extracted from text like:
        <tool>{"name": "gmail", "args": {"action": "read_inbox"}}</tool>
    """
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({"name": self.name, "args": self.arguments})

    @classmethod
    def from_json(cls, s: str) -> ToolCall:
        """Parse from JSON string."""
        data = json.loads(s)
        return cls(name=data["name"], arguments=data.get("args", {}))


# ─────────────────────────────────────────────
# ToolResult  (returned from tool execution)
# ─────────────────────────────────────────────
@dataclass
class ToolResult:
    """
    Result of executing a tool.

    Mirrors Gemma's mcp.types.CallToolResult pattern:
      - success: True if the tool executed without errors
      - output: the tool's text output (or error message)
    """
    name: str
    success: bool
    output: str

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "name": self.name,
            "success": self.success,
            "output": self.output,
        })


# ─────────────────────────────────────────────
# Tool ABC  (mirrors Gemma ToolHandlerBase)
# ─────────────────────────────────────────────
class Tool(abc.ABC):
    """
    Abstract base class for all tools.

    Subclasses must implement:
      - name (property): unique tool identifier
      - description (property): human-readable description for system prompt
      - parameters_schema (property): JSON Schema dict describing arguments
      - execute(**kwargs) → ToolResult: run the tool

    Usage:
        class MyTool(Tool):
            @property
            def name(self) -> str: return "my_tool"

            @property
            def description(self) -> str: return "Does something useful"

            @property
            def parameters_schema(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "search query"}
                    },
                    "required": ["query"]
                }

            def execute(self, **kwargs) -> ToolResult:
                return ToolResult(name=self.name, success=True, output="result")
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique tool identifier (used in tool calls)."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description (included in system prompt)."""
        ...

    @property
    @abc.abstractmethod
    def parameters_schema(self) -> dict:
        """
        JSON Schema dict describing the tool's arguments.

        Example:
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["read", "send"],
                        "description": "Action to perform"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["action"]
            }
        """
        ...

    @abc.abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given arguments.

        Args:
            **kwargs: tool-specific arguments matching parameters_schema

        Returns:
            ToolResult with success status and output text
        """
        ...

    def to_prompt_description(self) -> str:
        """
        Format this tool for inclusion in the system prompt.

        Generates a structured description that helps the model
        understand how to call this tool.
        """
        schema_str = json.dumps(self.parameters_schema, indent=2)
        return (
            f"Tool: {self.name}\n"
            f"Description: {self.description}\n"
            f"Parameters:\n{schema_str}\n"
            f"Usage: <tool>{{\"name\": \"{self.name}\", \"args\": {{...}}}}</tool>"
        )

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}')"
