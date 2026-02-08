"""Shared definitions for human-in-the-loop interactions."""

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    """What the user chose - unified vocabulary for logic and logging."""

    ACCEPT = "accept"
    REJECT = "reject"
    EDIT = "edit"
    SKIP = "skip"
    FEEDBACK = "feedback"


@dataclass(frozen=True)
class Option:
    """A user-selectable option in an interaction."""

    action: Action
    label: str  # Human-readable label for display


@dataclass(frozen=True)
class Interaction:
    """Definition of a human interaction point.

    Attributes:
        name: Identifier for logging (e.g., "check_code")
        title: Display title
        options: Available choices
        default: Default action if user doesn't choose
    """

    name: str
    title: str
    options: tuple[Option, ...]
    default: Action


# Interaction Definitions

CHECK_CODE = Interaction(
    name="check_code",
    title="Code found in response",
    options=(
        Option(Action.ACCEPT, "Check the code"),
        Option(Action.FEEDBACK, "Give feedback and regenerate response"),
        Option(Action.EDIT, "Edit the code manually"),
        Option(Action.SKIP, "Skip code check"),
    ),
    default=Action.ACCEPT,
)

ON_ERROR = Interaction(
    name="on_error",
    title="Code check failed",
    options=(
        Option(Action.ACCEPT, "Try to fix the code"),
        Option(
            Action.FEEDBACK, "Give extra feedback to the model on what might be wrong"
        ),
        Option(Action.SKIP, "Skip code fixing"),
    ),
    default=Action.ACCEPT,
)

RAG_QUERY = Interaction(
    name="rag_query",
    title="RAG Query Review",
    options=(
        Option(Action.ACCEPT, "Accept the query as-is"),
        Option(Action.EDIT, "Edit the query"),
        Option(Action.SKIP, "Skip retrieval completely"),
    ),
    default=Action.ACCEPT,
)

RAG_DOCS = Interaction(
    name="rag_docs",
    title="Modify retrieved documents",
    options=(
        Option(Action.ACCEPT, "Accept all documents"),
        Option(Action.EDIT, "Review and filter documents"),
        Option(Action.REJECT, "Reject all documents"),
    ),
    default=Action.ACCEPT,
)

TERMINAL_RUN = Interaction(
    name="terminal_run",
    title="Terminal Command Review",
    options=(
        Option(Action.ACCEPT, "Accept and run the command"),
        Option(Action.EDIT, "Edit the command and run"),
        Option(Action.SKIP, "Not run the command at all"),
    ),
    default=Action.SKIP,  # Safe default - don't run commands by default
)

FILE_WRITE = Interaction(
    name="file_write",
    title="File Write Confirmation",
    options=(
        Option(Action.ACCEPT, "Overwrite the file"),
        Option(Action.EDIT, "Specify a new path"),
        Option(Action.SKIP, "Skip writing"),
    ),
    default=Action.ACCEPT,
)
