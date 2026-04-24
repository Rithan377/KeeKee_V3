"""
registry.py — Tool Registry
===============================
Gemma-inspired (tools/_manager.py ToolHandlerBase.tools + call_tool):
  - Register tools by name
  - Dispatch ToolCall → Tool.execute() → ToolResult
  - Build system prompt section listing all available tools

Usage:
    registry = ToolRegistry()
    registry.register(GmailTool())
    result = registry.dispatch(ToolCall(name="gmail", arguments={...}))
    prompt_section = registry.system_prompt_section()
"""

from __future__ import annotations

from typing import Optional

from tools.base import Tool, ToolCall, ToolResult


# ─────────────────────────────────────────────
# ToolRegistry
# ─────────────────────────────────────────────
class ToolRegistry:
    """
    Central registry for all available tools.

    Provides:
      - register(tool): add a tool by its .name
      - get(name): look up a tool
      - dispatch(call): execute a ToolCall → ToolResult
      - system_prompt_section(): formatted string for system prompt

    Mirrors Gemma's ToolHandlerBase.tools property and call_tool() method.
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool. Raises ValueError if name already registered.

        Args:
            tool: Tool instance to register
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                f"Use a unique name for each tool."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """
        Look up a tool by name.

        Args:
            name: tool name

        Returns:
            Tool instance, or None if not found
        """
        return self._tools.get(name)

    def dispatch(self, call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Looks up the tool by name and calls execute() with the arguments.
        Returns an error ToolResult if the tool is not found or execution fails.

        Args:
            call: ToolCall with name and arguments

        Returns:
            ToolResult from the tool's execution
        """
        tool = self._tools.get(call.name)

        if tool is None:
            return ToolResult(
                name=call.name,
                success=False,
                output=f"Error: Tool '{call.name}' not found. "
                       f"Available tools: {', '.join(self.tool_names)}",
            )

        try:
            return tool.execute(**call.arguments)
        except Exception as e:
            return ToolResult(
                name=call.name,
                success=False,
                output=f"Error executing tool '{call.name}': {type(e).__name__}: {e}",
            )

    def system_prompt_section(self) -> str:
        """
        Build the tool descriptions section for the system prompt.

        Returns a formatted string describing all registered tools,
        their parameters, and how to call them.

        Returns:
            Formatted tool description string (empty if no tools)
        """
        if not self._tools:
            return ""

        lines = [
            "You have access to the following tools. To use a tool, "
            "output a tool call in this exact format:",
            '<tool>{"name": "tool_name", "args": {"param": "value"}}</tool>',
            "",
            "Available tools:",
            "",
        ]

        for tool in self._tools.values():
            lines.append(tool.to_prompt_description())
            lines.append("")

        return "\n".join(lines)

    @property
    def tool_names(self) -> list[str]:
        """List of registered tool names."""
        return list(self._tools.keys())

    @property
    def size(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names})"
