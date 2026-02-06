"""Define the configurable parameters for the agent."""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated, Optional, Type, TypeVar

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, ConfigDict

from jutulgpt import prompts

# Load configuration from TOML file.
PROJECT_ROOT = Path(__file__).resolve().parent
_TOML_PATH = PROJECT_ROOT.parent.parent / "jutulgpt.toml"


def _load_toml() -> dict:
    if _TOML_PATH.exists():
        with open(_TOML_PATH, "rb") as f:
            return tomllib.load(f)
    return {}


_toml = _load_toml()

# Static settings.
cli_mode: bool = _toml.get("mode", {}).get("cli", True)
mcp_mode: bool = _toml.get("mode", {}).get("mcp", False)
assert not (cli_mode and mcp_mode), "cli_mode and mcp_mode cannot both be true."

_DEFAULT_LLM: str = _toml.get("models", {}).get("llm", "openai:gpt-4.1")
LLM_TEMPERATURE: int = _toml.get("models", {}).get("temperature", 0)
RECURSION_LIMIT: int = _toml.get("agent", {}).get("recursion_limit", 200)


# Side-effect initialization (env vars, logging) — call explicitly before first API use.
_initialized = False


def init():
    """Initialize environment: load .env, set API keys, configure logging."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    import getpass

    from dotenv import load_dotenv

    load_dotenv()
    for var in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        if not os.environ.get(var):
            os.environ[var] = getpass.getpass(f"{var}: ")

    logging.getLogger("httpx").setLevel(logging.WARNING)


class HumanInteraction(BaseModel):
    model_config = ConfigDict(extra="forbid")  # optional strictness
    rag_query: bool = field(
        default=False,
        metadata={"description": "Whether to modify the generated RAG query."},
    )
    retrieved_examples: bool = field(
        default=False,
        metadata={
            "description": "Whether to verify and filter the retrieved examples."
        },
    )
    code_check: bool = field(
        default=True,
        metadata={
            "description": "Whether to perform code checks on the generated code."
        },
    )
    fix_error: bool = field(
        default=True,
        metadata={
            "description": "Whether to decide to try to fix errors in the generated code."
        },
    )


@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration class for the agent.

    This class defines the parameters needed for configuring the agent,
    including retrieval, model selection, and prompts.
    """

    # Human in the loop
    human_interaction: HumanInteraction = field(
        default_factory=HumanInteraction,
        metadata={
            "description": "Configuration for human interaction during the process. "
            "This includes options for RAG queries, retrieved documents, code checks, and multi-agent saving."
        },
    )

    # RAG
    retrieval_top_k: int = field(
        default_factory=lambda: _toml.get("retrieval", {}).get("top_k", 3),
        metadata={"description": "Number of documents to retrieve per query."},
    )

    # Models
    agent_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: _DEFAULT_LLM,
        metadata={"description": "The language model used for coding tasks."},
    )
    autonomous_agent_model: Annotated[
        str, {"__template_metadata__": {"kind": "llm"}}
    ] = field(
        default_factory=lambda: _DEFAULT_LLM,
        metadata={"description": "The language model used for coding tasks."},
    )

    # Prompts
    agent_prompt: str = field(
        default=prompts.AGENT_PROMPT,
        metadata={"description": "The default prompt used for the agent."},
    )
    autonomous_agent_prompt: str = field(
        default=prompts.AUTONOMOUS_AGENT_PROMPT,
        metadata={
            "description": "The default prompt used for the fully autonomous agent."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=BaseConfiguration)
