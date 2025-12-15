"""Conversation summarization for context management."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.language_models.base import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

from jutulgpt.configuration import SUMMARY_MSG_LIMIT
from jutulgpt.prompts import CONTEXT_SUMMARY_SYSTEM_PROMPT


def summarize_conversation(
    messages: List[BaseMessage],
    model: Union["BaseChatModel", "Runnable[LanguageModelInput, BaseMessage]"],
    config: "RunnableConfig",
    previous_summary: Optional[str] = None,
) -> Optional[str]:
    """Summarize the provided messages.

    Args:
        messages: Messages to summarize (conversation messages only).
        model: LLM for summarization (any invokable model).
        config: Runnable config.
        previous_summary: Previous summary to incorporate (if re-summarizing).

    Returns:
        Summary string, or None if empty/failed.
    """
    if not messages:
        return None

    # Build system prompt, incorporating previous summary if present
    system_content = CONTEXT_SUMMARY_SYSTEM_PROMPT
    if previous_summary:
        system_content += f"""

## Previous summary (from earlier in this conversation):
{previous_summary}

IMPORTANT: Your new summary must cover the ENTIRE conversation, not just the recent messages above.
Preserve the key topics and outcomes from the previous summary."""

    # Truncate messages to control summarization cost
    truncated = [_truncate_message(msg, SUMMARY_MSG_LIMIT) for msg in messages]

    # Invoke model with messages directly (native role handling)
    response = model.invoke(
        [
            SystemMessage(content=system_content),
            *truncated,
            HumanMessage(content="Please provide the summary now."),
        ],
        config,
    )

    summary = response.content if isinstance(response.content, str) else str(response.content)
    return summary if summary and len(summary) > 50 else None


def _truncate_message(msg: BaseMessage, char_limit: int) -> BaseMessage:
    """Return a copy of the message with truncated content if needed."""
    content = msg.content if isinstance(msg.content, str) else str(msg.content)

    if len(content) <= char_limit:
        return msg

    truncated_content = content[:char_limit] + "... [truncated]"

    return msg.model_copy(update={"content": truncated_content})
