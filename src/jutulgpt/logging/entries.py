"""Typed log entry dataclasses for structured session logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from jutulgpt.human_in_the_loop.interactions import Interaction


class EntryType(str, Enum):
    """Categories of log entries for filtering and formatting."""

    ASSISTANT = "assistant"
    TOOL = "tool"
    USER = "user"
    SYSTEM = "system"
    RAG = "rag"
    INTERACTION = "interaction"


@dataclass
class LogEntry:
    """Base class for all log entries.

    Attributes:
        content: The main text content to log.
        title: Display title for the entry (e.g., agent name, tool name).
        entry_type: Category of the entry for filtering/formatting.
        timestamp: When the entry was created (auto-generated if not provided).
    """

    content: str
    title: str = "JutulGPT"
    entry_type: EntryType = EntryType.ASSISTANT
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # Ensure content is always a string
        if self.content is None:
            self.content = ""
        elif not isinstance(self.content, str):
            self.content = str(self.content)


@dataclass
class AssistantEntry(LogEntry):
    """Log entry for assistant/agent responses.

    Attributes:
        tool_calls: List of tool calls made by the assistant.
        config: Filtered configuration used for the response.
    """

    entry_type: EntryType = field(default=EntryType.ASSISTANT, init=False)
    tool_calls: Optional[list[dict[str, Any]]] = None
    config: Optional[dict[str, Any]] = None
    reasoning_summary: Optional[str] = None


@dataclass
class ToolEntry(LogEntry):
    """Log entry for tool executions.

    Attributes:
        tool_name: Name of the tool that was executed.
        args: Arguments passed to the tool.
        returncode: Exit code (for command execution).
        error: Error message if the tool failed.
    """

    entry_type: EntryType = field(default=EntryType.TOOL, init=False)
    tool_name: str = ""
    args: Optional[dict[str, Any]] = None
    returncode: Optional[int] = None
    error: Optional[str] = None


@dataclass
class UserEntry(LogEntry):
    """Log entry for user inputs."""

    entry_type: EntryType = field(default=EntryType.USER, init=False)


@dataclass
class RAGEntry(LogEntry):
    """Log entry for RAG retrieval operations.

    Attributes:
        query: The search query used.
        source: Which retriever/store was queried.
        num_results: Number of results returned.
    """

    entry_type: EntryType = field(default=EntryType.RAG, init=False)
    query: Optional[str] = None
    source: Optional[str] = None
    num_results: Optional[int] = None


@dataclass
class CodeRunnerEntry(LogEntry):
    """Log entry for Julia code execution.

    Attributes:
        code: The Julia code that was executed.
        language: Programming language (default: julia).
        success: Whether execution succeeded.
    """

    entry_type: EntryType = field(default=EntryType.TOOL, init=False)
    tool_name: str = "Code Runner"
    code: Optional[str] = None
    language: str = "julia"
    success: Optional[bool] = None


@dataclass
class InteractionEntry(LogEntry):
    """Log entry for human-in-the-loop interactions.

    Attributes:
        content: The selected option label (e.g., "Check the code").
        title: The interaction title (e.g., "Code found in response").
        interaction: The interaction definition for displaying options.
        action: The action the user chose (e.g., "accept").
        user_input: Additional text input provided by user (for feedback/edits).
    """

    entry_type: EntryType = field(default=EntryType.INTERACTION, init=False)
    interaction: "Interaction | None" = None
    action: str = ""
    user_input: Optional[str] = None
