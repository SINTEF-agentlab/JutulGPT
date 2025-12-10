"""Markdown formatting utilities for session logs.

These functions are pure and testable - they take data and return strings.
"""

from __future__ import annotations

import json
from typing import Any, Optional

# Default limits for content truncation
DEFAULT_CONTENT_MAX_LEN = 2000
DEFAULT_ARGS_MAX_LEN = 1200


def truncate(text: str, max_len: int = DEFAULT_CONTENT_MAX_LEN) -> str:
    """Truncate text to max_len, adding ellipsis if truncated."""
    if not text or len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_json(data: Any, max_len: int = DEFAULT_ARGS_MAX_LEN) -> str:
    """Format data as indented JSON, truncating if needed."""
    try:
        rendered = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    except Exception:
        rendered = str(data)
    return truncate(rendered, max_len)


def format_tool_calls(
    tool_calls: list[dict[str, Any]],
    max_args_len: int = DEFAULT_ARGS_MAX_LEN,
) -> str:
    """Format tool calls for the log.

    Args:
        tool_calls: List of tool call dicts with 'name' and 'args'.
        max_args_len: Maximum length for serialized arguments.

    Returns:
        Formatted markdown string.
    """
    if not tool_calls:
        return ""

    lines = []
    for call in tool_calls:
        name = call.get("name", "unknown_tool")
        args = call.get("args", {})
        args_str = format_json(args, max_args_len)
        lines.append(f"**`{name}`**")
        lines.append(f"```json\n{args_str}\n```")
        lines.append("")

    return "\n".join(lines)


def format_config(config: dict[str, Any]) -> str:
    """Format configuration as a collapsible details block."""
    if not config:
        return ""

    config_json = format_json(config, max_len=2000)
    return f"<details><summary>Configuration</summary>\n\n```json\n{config_json}\n```\n</details>"


def format_code_block(code: str, language: str = "") -> str:
    """Wrap code in a fenced code block."""
    return f"```{language}\n{code}\n```"


def format_tool_args(args: Optional[dict[str, Any]]) -> str:
    """Format tool arguments inline or as a block depending on size."""
    if not args:
        return ""

    # For small args, show inline
    simple_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
    if len(simple_str) < 80:
        return f"`{simple_str}`"

    # For larger args, use JSON block
    return f"```json\n{format_json(args)}\n```"
