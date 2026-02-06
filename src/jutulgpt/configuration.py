"""Define the configurable parameters for the agent."""

from __future__ import annotations

import getpass
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated, Optional, Type, TypeVar

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, ConfigDict

from jutulgpt import prompts

# Static settings.
# NOTE: Currently only one of these can be true at a time
cli_mode: bool = True  # If the agent is run from using the CLI
mcp_mode: bool = (
    False  # If the agent is run as an MPC server that can be called from VSCode
)
assert not (cli_mode and mcp_mode), "cli_mode and mcp_mode cannot both be true."

# Select whether to use local models through Ollama or use OpenAI
LOCAL_MODELS = False
LLM_MODEL_NAME = "ollama:qwen3:14b" if LOCAL_MODELS else "openai:gpt-4.1"
RECURSION_LIMIT = 200  # Number of recursions before an error is thrown.
LLM_TEMPERATURE = 0


# Setup of the environment and some logging. Not neccessary to touch this.
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv()
_set_env("OPENAI_API_KEY")
_set_env("LANGSMITH_API_KEY")


logging.getLogger("httpx").setLevel(logging.WARNING)  # Less warnings in the output


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
        default=3,
        metadata={"description": "Number of documents to retrieve per query."},
    )

    # Models
    agent_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: LLM_MODEL_NAME,
        metadata={"description": "The language model used for coding tasks."},
    )
    autonomous_agent_model: Annotated[
        str, {"__template_metadata__": {"kind": "llm"}}
    ] = field(
        default_factory=lambda: LLM_MODEL_NAME,
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
