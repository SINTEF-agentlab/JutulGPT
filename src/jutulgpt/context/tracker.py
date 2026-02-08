"""Context usage tracking for the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from jutulgpt.logging import SessionLogger

from rich.text import Text

from jutulgpt.configuration import CONTEXT_DISPLAY_THRESHOLD, MODEL_CONTEXT_WINDOW
from jutulgpt.globals import console

# Colors/styles for different context categories
# Chosen to be: 1) visually distinct, 2) work on both light and dark terminals
# TODO: Check that all colors in the application work on both light and dark terminals
COLORS = {
    "system": "blue",
    "summary": "magenta",
    "tool_output": "yellow",
    "messages": "green",
    "free": "dim",
}


@dataclass
class ContextUsage:
    """Token usage breakdown by category."""

    system: int = 0  # System prompt + workspace + tool definitions
    summary: int = 0  # Context summary (if active)
    tool_output: int = 0  # ToolMessages + tool_calls
    messages: int = 0  # Human + AI content
    total: int = 0
    max_tokens: int = MODEL_CONTEXT_WINDOW

    @property
    def free_space(self) -> int:
        return max(0, self.max_tokens - self.total)

    @property
    def usage_fraction(self) -> float:
        return self.total / self.max_tokens if self.max_tokens > 0 else 0.0

    @property
    def usage_percent(self) -> float:
        return self.usage_fraction * 100


def _estimate_tokens(content: str, model: Optional["BaseChatModel"] = None) -> int:
    """Estimate tokens: use model tokenizer if available, else ~4 chars/token."""
    if not content:
        return 0
    if model is not None:
        try:
            return model.get_num_tokens(content)
        except Exception:
            pass
    return len(content) // 4


class ContextTracker:
    """Tracks and displays context token usage.

    Static context (system prompt, tools) is counted once at init.
    Messages are counted on each update.
    """

    def __init__(
        self,
        system_prompt: str = "",
        tool_definitions: Optional[List] = None,
        model: Optional["BaseChatModel"] = None,
        max_tokens: int = MODEL_CONTEXT_WINDOW,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._last_usage: Optional[ContextUsage] = None

        # Count static context once (prompt + tools combined)
        self._system_tokens = _estimate_tokens(system_prompt, model)
        if tool_definitions:
            self._system_tokens += _estimate_tokens(str(tool_definitions), model)

    def get_last_usage(self) -> Optional[ContextUsage]:
        return self._last_usage

    def update(
        self, messages: Sequence[BaseMessage], summary: str = ""
    ) -> ContextUsage:
        """Update usage with current messages. Returns usage breakdown.

        Args:
            messages: Conversation messages (excludes system prompt/workspace).
            summary: The context summary text (if active).
        """
        usage = ContextUsage(
            system=self._system_tokens,
            summary=_estimate_tokens(summary, self._model) if summary else 0,
            max_tokens=self._max_tokens,
        )

        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            tokens = _estimate_tokens(content, self._model)

            if isinstance(msg, ToolMessage):
                usage.tool_output += tokens
            elif isinstance(msg, (HumanMessage, AIMessage)):
                usage.messages += tokens
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    usage.tool_output += _estimate_tokens(
                        str(msg.tool_calls), self._model
                    )

        usage.total = usage.system + usage.summary + usage.tool_output + usage.messages
        self._last_usage = usage
        return usage

    def display(
        self,
        logger: Optional["SessionLogger"] = None,
        show_console: bool = True,
        console_threshold: float = CONTEXT_DISPLAY_THRESHOLD,
    ) -> None:
        """Display context usage to console and/or log file.

        Args:
            logger: If provided, always writes to log file.
            show_console: Whether to show in console at all.
            console_threshold: Only show in console if usage exceeds this fraction.

        Principle: Wherever we print to console, we also log.
        """
        usage = self._last_usage
        if not usage:
            return

        # Always log if logger provided
        if logger and logger.enabled:
            logger._write_raw(self._format_markdown(usage))

        # Show in console if enabled and above threshold
        if show_console and usage.usage_fraction >= console_threshold:
            self._display_console(usage)

    def _format_markdown(self, usage: ContextUsage) -> str:
        """Format context usage as markdown for logging."""

        def fmt(n: int) -> str:
            return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

        def pct(n: int) -> str:
            p = n / usage.max_tokens * 100 if usage.max_tokens else 0
            return f"({p:.1f}%)"

        bar_width = 40
        filled = int(bar_width * usage.usage_fraction)

        lines = [
            "#### Context Usage\n\n",
            "```\n",
            f"{'█' * filled}{'░' * (bar_width - filled)} "
            f"{fmt(usage.total)}/{fmt(usage.max_tokens)} ({usage.usage_percent:.1f}%)\n",
            f"├─ System:          {fmt(usage.system):>8} {pct(usage.system):>8}\n",
        ]
        if usage.summary > 0:
            lines.append(
                f"├─ Context summary: {fmt(usage.summary):>8} {pct(usage.summary):>8}\n"
            )
        lines.extend(
            [
                f"├─ Tool output:     {fmt(usage.tool_output):>8} {pct(usage.tool_output):>8}\n",
                f"├─ Messages:        {fmt(usage.messages):>8} {pct(usage.messages):>8}\n",
                f"└─ Free space:      {fmt(usage.free_space):>8} {pct(usage.free_space):>8}\n",
                "```\n\n---\n\n",
            ]
        )
        return "".join(lines)

    def _display_console(self, usage: ContextUsage) -> None:
        """Display context usage to console with Rich formatting."""
        # Overall color based on usage level
        if usage.usage_fraction > 0.85:
            bar_color = "red"
        elif usage.usage_fraction > 0.7:
            bar_color = "yellow"
        else:
            bar_color = "cyan"

        def fmt(n: int) -> str:
            return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

        def pct(n: int) -> str:
            p = n / usage.max_tokens * 100 if usage.max_tokens else 0
            return f"({p:.1f}%)"

        # Build colored progress bar (segments for each category)
        bar_width = 40
        sys_w = (
            int(bar_width * usage.system / usage.max_tokens) if usage.max_tokens else 0
        )
        sum_w = (
            int(bar_width * usage.summary / usage.max_tokens) if usage.max_tokens else 0
        )
        tool_w = (
            int(bar_width * usage.tool_output / usage.max_tokens)
            if usage.max_tokens
            else 0
        )
        msg_w = (
            int(bar_width * usage.messages / usage.max_tokens)
            if usage.max_tokens
            else 0
        )
        free_w = bar_width - (sys_w + sum_w + tool_w + msg_w)

        # Aligned output using Text objects for proper style boundaries
        lw, vw, pw = 17, 8, 8  # label, value, percent widths

        def make_row(branch: str, label: str, color: str, value: int) -> Text:
            """Build a row with proper style isolation."""
            row = Text()
            row.append("  ")
            row.append(branch, style="dim")
            row.append(" ")
            row.append(f"{label:<{lw}}", style=color)
            row.append(f"{fmt(value):>{vw}} {pct(value):>{pw}}")
            return row

        # Header
        console.print()
        console.print(
            f"[bold {bar_color}]Context Usage[/bold {bar_color}]", justify="center"
        )

        # Bar line
        bar_line = Text()
        bar_line.append("  ")
        bar_line.append("█" * sys_w, style=COLORS["system"])
        bar_line.append("█" * sum_w, style=COLORS["summary"])
        bar_line.append("█" * tool_w, style=COLORS["tool_output"])
        bar_line.append("█" * msg_w, style=COLORS["messages"])
        bar_line.append("░" * free_w, style="dim")
        bar_line.append(
            f" {fmt(usage.total)}/{fmt(usage.max_tokens)} ({usage.usage_percent:.1f}%)"
        )
        console.print(bar_line)

        # Detail rows
        console.print(make_row("├─", "System:", COLORS["system"], usage.system))
        if usage.summary > 0:
            console.print(
                make_row("├─", "Context summary:", COLORS["summary"], usage.summary)
            )
        console.print(
            make_row("├─", "Tool output:", COLORS["tool_output"], usage.tool_output)
        )
        console.print(make_row("├─", "Messages:", COLORS["messages"], usage.messages))
        console.print(make_row("└─", "Free space:", COLORS["free"], usage.free_space))
        console.print()
