"""Model loading and message utilities."""

from typing import List, Sequence, Union

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.runnables import Runnable

from jutulgpt import configuration as cfg
from jutulgpt.configuration import ModelConfig


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def get_provider_and_model(name: str) -> tuple[str, str]:
    """
    Get the provider and name from a string on the format 'provider:model'.
    """
    provider, model = name.split(":", maxsplit=1)
    return provider, model


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider:model'.
    """
    provider, model = get_provider_and_model(fully_specified_name)
    # Prefer the active preset when it matches the requested provider/model.
    # Otherwise, fall back to a minimal config for that provider/model.
    active_cfg = cfg.ACTIVE_MODEL_CONFIG
    if active_cfg.provider == provider and active_cfg.model == model:
        model_cfg = active_cfg
    else:
        model_cfg = ModelConfig(
            provider=provider,  # type: ignore[arg-type]
            model=model,
        )

    try:
        return init_chat_model(
            model_cfg.model,
            model_provider=model_cfg.provider,
            streaming=True,
            **model_cfg.llm_kwargs,
        )
    except Exception as e:
        hint = ""
        msg = str(e)
        if provider == "ollama":
            # Common Ollama failure mode: model not pulled locally.
            if "not found" in msg and "pull" in msg:
                hint = f" If using Ollama, try: `ollama pull {model}`."

        raise ValueError(
            f"Failed to load chat model '{provider}:{model}': {e}. "
            "Ensure the model name is correct, credentials are set (if required), "
            "and the provider runtime is available." + hint
        ) from e


def trim_state_messages(
    messages: Sequence[BaseMessage],
    model: Union[BaseChatModel, Runnable[LanguageModelInput, BaseMessage]],
) -> Sequence[BaseMessage]:
    trimmed_state_messages = trim_messages(
        messages,
        max_tokens=40000,  # adjust for model's context window minus system & files message
        strategy="last",
        token_counter=model,
        include_system=False,  # Not needed since systemMessage is added separately
        allow_partial=True,
    )
    return trimmed_state_messages


def get_tool_message(messages: List, n_last=2, print=False):
    """
    Extract the most recent tool message from the last n messages.

    Args:
        messages (list): List of message objects.
        n_last (int): Number of last messages to consider.
        print (bool): If True, pretty print the found message.

    Returns:
        The most recent tool message object, or None if not found.
    """
    for message in messages[-n_last:]:
        if message.type == "tool":
            if print:
                message.pretty_print()
            return message
    return None
