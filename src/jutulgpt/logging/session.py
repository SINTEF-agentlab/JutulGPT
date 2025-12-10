"""Session logger for writing markdown conversation logs.

The SessionLogger class manages a single log file per session and provides
typed methods for logging different kinds of entries.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from jutulgpt.julia.version_info import get_version_info
from jutulgpt.logging.entries import (
    AssistantEntry,
    CodeRunnerEntry,
    LogEntry,
    RAGEntry,
    ToolEntry,
    UserEntry,
)
from jutulgpt.logging.formatters import (
    format_code_block,
    format_config,
    format_tool_args,
    format_tool_calls,
)

if TYPE_CHECKING:
    from jutulgpt.configuration import BaseConfiguration


class SessionLogger:
    """Manages session logging to a markdown file.

    Usage:
        logger = SessionLogger.from_config(configuration)
        logger.log(AssistantEntry(content="Hello!", title="Agent"))
        logger.log(ToolEntry(content="Output", tool_name="run_julia_code", args={...}))
    """

    def __init__(
        self,
        log_path: Path,
        enabled: bool = True,
        session_title: Optional[str] = None,
        include_version_info: bool = True,
    ):
        """Initialize the session logger.

        Args:
            log_path: Path to the markdown log file.
            enabled: Whether logging is active.
            session_title: Optional title for the session header.
            include_version_info: Whether to include Julia/Jutul/JutulDarcy versions.
        """
        self._log_path = log_path
        self._enabled = enabled
        self._session_title = session_title or "JutulGPT Session"
        self._include_version_info = include_version_info
        self._initialized = False

    @classmethod
    def from_config(cls, config: "BaseConfiguration") -> "SessionLogger":
        """Create a SessionLogger from a BaseConfiguration instance.

        Args:
            config: The agent configuration object.

        Returns:
            Configured SessionLogger instance.
        """
        if not config.log_to_file:
            return cls(log_path=Path("/dev/null"), enabled=False)

        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{config.log_filename_prefix}_{timestamp}.md"
        log_path = log_dir / filename

        # Get version info setting from config if available
        include_version_info = getattr(config, "log_version_info", True)

        return cls(log_path=log_path, enabled=True, include_version_info=include_version_info)

    @property
    def enabled(self) -> bool:
        """Whether logging is currently enabled."""
        return self._enabled

    @property
    def log_path(self) -> Path:
        """Path to the current log file."""
        return self._log_path

    def _ensure_initialized(self) -> None:
        """Write session header on first log entry."""
        if self._initialized or not self._enabled:
            return

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        header = self._format_session_header()
        self._write_raw(header)
        self._initialized = True

    def _format_session_header(self) -> str:
        """Generate the session header markdown."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"# {self._session_title}\n\n"]
        lines.append(f"_Session started: {timestamp}_\n")

        # Add version info if enabled
        if self._include_version_info:
            version_info = get_version_info()
            if version_info:
                lines.append(f"\n_{version_info.format_markdown()}_\n")

        lines.append("\n---\n\n")
        return "".join(lines)

    def _write_raw(self, content: str) -> None:
        """Append raw content to the log file."""
        if not self._enabled:
            return

        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(content)
        except Exception as exc:
            # Fail silently to avoid crashing the agent
            import sys

            print(f"Warning: Failed to write to log '{self._log_path}': {exc}", file=sys.stderr)

    def log(self, entry: LogEntry) -> None:
        """Log a typed entry to the session file.

        Args:
            entry: A LogEntry subclass instance (AssistantEntry, ToolEntry, etc.)
        """
        if not self._enabled:
            return

        self._ensure_initialized()

        # Dispatch to specific formatter based on entry type
        if isinstance(entry, AssistantEntry):
            formatted = self._format_assistant_entry(entry)
        elif isinstance(entry, ToolEntry):
            formatted = self._format_tool_entry(entry)
        elif isinstance(entry, CodeRunnerEntry):
            formatted = self._format_code_runner_entry(entry)
        elif isinstance(entry, UserEntry):
            formatted = self._format_user_entry(entry)
        elif isinstance(entry, RAGEntry):
            formatted = self._format_rag_entry(entry)
        else:
            formatted = self._format_generic_entry(entry)

        self._write_raw(formatted)

    def _format_entry_header(self, entry: LogEntry) -> str:
        """Format the common entry header."""
        timestamp = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"## {entry.title}\n\n_{timestamp}_\n\n"

    def _format_assistant_entry(self, entry: AssistantEntry) -> str:
        """Format an assistant response entry."""
        lines = [f"## {entry.title}\n\n"]

        # Main content
        if entry.content:
            lines.append(f"{entry.content}\n")

        # Tool calls section
        if entry.tool_calls:
            lines.append("\n### Tool Calls\n")
            lines.append(format_tool_calls(entry.tool_calls))

        # Config section (if provided)
        if entry.config:
            lines.append("\n")
            lines.append(format_config(entry.config))

        lines.append("\n---\n\n")
        return "".join(lines)

    def _format_tool_entry(self, entry: ToolEntry) -> str:
        """Format a tool execution entry."""
        lines = [f"#### {entry.title}"]
        if entry.tool_name:
            lines.append(f" — `{entry.tool_name}`")
        lines.append("\n\n")

        # Arguments
        if entry.args:
            args_formatted = format_tool_args(entry.args)
            # If args contain a code fence, put on new line
            if "```" in args_formatted:
                lines.append(f"**Args:**\n{args_formatted}\n\n")
            else:
                lines.append(f"**Args:** {args_formatted}\n\n")

        # Output content
        if entry.content:
            lines.append(f"{entry.content}\n")

        # Status info
        status_parts = []
        if entry.returncode is not None:
            status_parts.append(f"exit code: {entry.returncode}")
        if entry.error:
            status_parts.append(f"error: {entry.error}")
        if status_parts:
            lines.append(f"\n_({', '.join(status_parts)})_\n")

        lines.append("\n---\n\n")
        return "".join(lines)

    def _format_code_runner_entry(self, entry: CodeRunnerEntry) -> str:
        """Format a code execution entry."""
        lines = [f"#### {entry.title}"]
        if entry.tool_name:
            lines.append(f" — `{entry.tool_name}`")
        lines.append("\n\n")

        # Show the code that was run
        if entry.code:
            lines.append("**Code:**\n")
            lines.append(format_code_block(entry.code, entry.language))
            lines.append("\n\n")

        # Output
        if entry.content:
            lines.append("**Output:**\n")
            lines.append(f"{entry.content}\n")

        # Success indicator
        if entry.success is not None:
            status = "✓ Success" if entry.success else "✗ Failed"
            lines.append(f"\n_{status}_\n")

        lines.append("\n---\n\n")
        return "".join(lines)

    def _format_user_entry(self, entry: UserEntry) -> str:
        """Format a user input entry."""
        lines = [f"## {entry.title}\n\n"]
        if entry.content:
            lines.append(f"{entry.content}\n")
        lines.append("\n---\n\n")
        return "".join(lines)

    def _format_rag_entry(self, entry: RAGEntry) -> str:
        """Format a RAG retrieval entry."""
        lines = [f"#### {entry.title}"]
        if entry.source:
            lines.append(f" — `{entry.source}`")
        lines.append("\n\n")

        if entry.query:
            lines.append(f"**Query:** {entry.query}\n\n")

        if entry.num_results is not None:
            lines.append(f"**Results:** {entry.num_results}\n\n")

        if entry.content:
            lines.append(f"{entry.content}\n")

        lines.append("\n---\n\n")
        return "".join(lines)

    def _format_generic_entry(self, entry: LogEntry) -> str:
        """Format a generic log entry."""
        lines = [self._format_entry_header(entry)]
        if entry.content:
            lines.append(f"{entry.content}\n")
        lines.append("\n---\n\n")
        return "".join(lines)

    # Convenience methods for common entry types

    def log_assistant(
        self,
        content: str,
        title: str = "JutulGPT",
        tool_calls: Optional[list] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Convenience method to log an assistant response."""
        self.log(
            AssistantEntry(
                content=content,
                title=title,
                tool_calls=tool_calls,
                config=config,
            )
        )

    def log_tool(
        self,
        content: str,
        tool_name: str,
        title: str = "Tool Output",
        args: Optional[dict] = None,
        returncode: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        """Convenience method to log a tool execution."""
        self.log(
            ToolEntry(
                content=content,
                title=title,
                tool_name=tool_name,
                args=args,
                returncode=returncode,
                error=error,
            )
        )

    def log_user(self, content: str, title: str = "User") -> None:
        """Convenience method to log user input."""
        self.log(UserEntry(content=content, title=title))

    def log_code_run(
        self,
        output: str,
        code: str,
        title: str = "Code Runner",
        language: str = "julia",
        success: Optional[bool] = None,
    ) -> None:
        """Convenience method to log code execution."""
        self.log(
            CodeRunnerEntry(
                content=output,
                title=title,
                code=code,
                language=language,
                success=success,
            )
        )

    def log_rag(
        self,
        content: str,
        query: str,
        source: str = "JutulDarcy",
        num_results: Optional[int] = None,
        title: str = "RAG Retrieval",
    ) -> None:
        """Convenience method to log RAG retrieval."""
        self.log(
            RAGEntry(
                content=content,
                title=title,
                query=query,
                source=source,
                num_results=num_results,
            )
        )


# Global session logger instance (initialized lazily)
_session_logger: Optional[SessionLogger] = None


def get_session_logger() -> Optional[SessionLogger]:
    """Get the current global session logger."""
    return _session_logger


def set_session_logger(logger: SessionLogger) -> None:
    """Set the global session logger."""
    global _session_logger
    _session_logger = logger


def init_session_logger(config: "BaseConfiguration") -> SessionLogger:
    """Initialize and set the global session logger from config.

    Args:
        config: The agent configuration.

    Returns:
        The initialized SessionLogger.
    """
    logger = SessionLogger.from_config(config)
    set_session_logger(logger)
    return logger
