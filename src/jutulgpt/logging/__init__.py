"""Session logging for JutulGPT conversations.

This module provides structured logging of agent conversations to markdown files
for debugging, sharing, and analysis.

Usage:
    from jutulgpt.logging import SessionLogger, AssistantEntry, ToolEntry

    # Initialize from configuration
    logger = SessionLogger.from_config(configuration)

    # Log an assistant response
    logger.log_assistant(
        content="Here's the code...",
        title="JutulGPT",
        tool_calls=[...],
    )

    # Log a tool execution
    logger.log_tool(
        content="stdout output...",
        tool_name="execute_terminal_command",
        args={"command": "julia script.jl"},
        returncode=0,
    )

    # Or use typed entries directly
    logger.log(AssistantEntry(content="...", title="Agent"))
"""

from jutulgpt.logging.entries import (
    AssistantEntry,
    CodeRunnerEntry,
    EntryType,
    LogEntry,
    RAGEntry,
    ToolEntry,
    UserEntry,
)
from jutulgpt.logging.formatters import (
    format_code_block,
    format_config,
    format_json,
    format_tool_args,
    format_tool_calls,
    truncate,
)
from jutulgpt.logging.session import (
    SessionLogger,
    get_session_logger,
    init_session_logger,
    set_session_logger,
)

__all__ = [
    # Entry types
    "LogEntry",
    "AssistantEntry",
    "ToolEntry",
    "UserEntry",
    "RAGEntry",
    "CodeRunnerEntry",
    "EntryType",
    # Session logger
    "SessionLogger",
    "get_session_logger",
    "set_session_logger",
    "init_session_logger",
    # Formatters (for custom use)
    "truncate",
    "format_json",
    "format_tool_calls",
    "format_config",
    "format_code_block",
    "format_tool_args",
]
