"""Define the configurable parameters for the agent."""

from __future__ import annotations

import getpass
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, Type, TypeVar

# Load environment before importing LangChain (it checks for tracing at import time)
from dotenv import load_dotenv

load_dotenv()

if os.environ.get("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_TRACING", "true")

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

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           MODEL SELECTION                                    ║
# ╠══════════════════════════════════════════════════════════════════════════════╣
# ║  Supported providers:                                                        ║
# ║    - ollama: Local models via Ollama                                         ║
# ║    - openai: OpenAI API via API key (requires API key in .env)               ║
# ║                                                                              ║
# ║  Notes:                                                                      ║
# ║    - OpenAI models default to the Responses API (recommended).               ║
# ║    - We still manage message/state ourselves in LangGraph/LangChain.         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@dataclass(frozen=True)
class ModelConfig:
    """Temporary model configuration container."""

    provider: Literal["ollama", "openai"]
    model: str
    # Context window size in tokens (used for context tracking + summarization thresholds)
    context_window: int = 200000
    # Provider-specific kwargs forwarded to `init_chat_model(...)`.
    # Examples:
    # - OpenAI: {"use_responses_api": True, "verbosity": "low", "reasoning": {"effort": "medium", "summary": "auto"}}
    # - Ollama: {"reasoning": True}
    llm_kwargs: dict[str, Any] = field(default_factory=dict)


# Explicit supported model configurations
OPENAI_GPT_4_1 = ModelConfig(
    provider="openai",
    model="gpt-4.1",
    context_window=200000,
    llm_kwargs={
        "temperature": 0.0,
        "use_responses_api": True,
        # gpt-4.1 currently only supports "medium" verbosity via OpenAI Responses API.
        "verbosity": "medium",
    },
)
OPENAI_GPT_5_MINI = ModelConfig(
    provider="openai",
    model="gpt-5-mini",
    context_window=200000,
    llm_kwargs={
        "temperature": 0.0,
        "use_responses_api": True,
        "verbosity": "low",
    },
)
OPENAI_GPT_5_MINI_REASONING = ModelConfig(
    provider="openai",
    model="gpt-5-mini",
    context_window=200000,
    llm_kwargs={
        "temperature": 0.0,
        "use_responses_api": True,
        "verbosity": "low",
        "reasoning": {"effort": "medium", "summary": "auto"},
    },
)
OPENAI_GPT_5_1 = ModelConfig(
    provider="openai",
    model="gpt-5.1",
    context_window=200000,
    llm_kwargs={
        "temperature": 0.0,
        "use_responses_api": True,
        "verbosity": "low",
    },
)
OPENAI_GPT_5_1_REASONING = ModelConfig(
    provider="openai",
    model="gpt-5.1",
    context_window=200000,
    llm_kwargs={
        "temperature": 0.0,
        "use_responses_api": True,
        "verbosity": "low",
        "reasoning": {"effort": "medium", "summary": "auto"},
    },
)
OPENAI_GPT_5_2 = ModelConfig(
    provider="openai",
    model="gpt-5.2",
    context_window=200000,
    llm_kwargs={
        "temperature": 0.0,
        "use_responses_api": True,
        "verbosity": "low",
    },
)
OPENAI_GPT_5_2_REASONING = ModelConfig(
    provider="openai",
    model="gpt-5.2",
    context_window=200000,
    llm_kwargs={
        "temperature": 0.0,
        "use_responses_api": True,
        "verbosity": "low",
        "reasoning": {"effort": "medium", "summary": "auto"},
    },
)
# Qwen3 via Ollama (thinking enabled; thoughts separated from content)
OLLAMA_QWEN3_14B_THINKING = ModelConfig(
    provider="ollama",
    model="qwen3:14b",
    context_window=32000,
    llm_kwargs={
        # Recommended: slightly lower temp for thinking / reasoning mode.
        "temperature": 0.6,
        "num_ctx": 32000,
        "reasoning": True,
    },
)

# Qwen3 via Ollama (thinking disabled)
OLLAMA_QWEN3_14B = ModelConfig(
    provider="ollama",
    model="qwen3:14b",
    context_window=32000,
    llm_kwargs={
        # Recommended: slightly higher temp for non-thinking chat mode.
        "temperature": 0.7,
        "num_ctx": 32000,
        "reasoning": False,
    },
)


# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  MODEL CONFIGURATION - Change these values to switch models                  │
# │  DEFAULT_MODEL_PRESET is used by the CLI when you omit --model.              │
# └──────────────────────────────────────────────────────────────────────────────┘
DEFAULT_MODEL_PRESET: str = "gpt-5.2-reasoning"
ACTIVE_MODEL_CONFIG: ModelConfig = OPENAI_GPT_5_2_REASONING

# Print/log the reasoning summary blocks (if returned by OpenAI)
SHOW_REASONING_SUMMARY: bool = True

ACTIVE_PROVIDER: Literal["ollama", "openai"] = ACTIVE_MODEL_CONFIG.provider
ACTIVE_MODEL_NAME: str = ACTIVE_MODEL_CONFIG.model

# String used elsewhere ("provider:model")
ACTIVE_MODEL: str = f"{ACTIVE_PROVIDER}:{ACTIVE_MODEL_NAME}"

# Embedding model (matches provider of active model)
_EMBEDDING_MODEL_BY_PROVIDER: dict[str, str] = {
    "ollama": "ollama:nomic-embed-text",
    "openai": "openai:text-embedding-3-small",
}
EMBEDDING_MODEL_NAME: str = _EMBEDDING_MODEL_BY_PROVIDER[ACTIVE_PROVIDER]

RECURSION_LIMIT = 200  # Number of recursions before an error is thrown.

# Display settings - for console and log output (not context management)
DISPLAY_CONTENT_MAX_LENGTH = 800  # Max chars to display in console/logs

# Context management settings
# ┌─────────────────────────────────────────────────────────────────┐
# │  MODEL_CONTEXT_WINDOW (active model context window)             │
# │  ├── System prompt + workspace + summary                        │
# │  ├── Tool definitions                                           │
# │  ├── Tool results (ToolMessages + tool_calls)                   │
# │  └── Messages (Human + AI content)                              │
# │                                                                 │
# │  Thresholds (fractions of MODEL_CONTEXT_WINDOW):                │
# │  ├── CONTEXT_USAGE_THRESHOLD (0.7) → trigger summarization      │
# │  └── CONTEXT_TRIM_THRESHOLD (0.9)  → safety trim if needed      │
# └─────────────────────────────────────────────────────────────────┘
MODEL_CONTEXT_WINDOW = ACTIVE_MODEL_CONFIG.context_window  # Total context budget in tokens
CONTEXT_USAGE_THRESHOLD = 0.7  # Summarization trigger threshold
CONTEXT_TRIM_THRESHOLD = 0.9  # Safety trim threshold (if summarization didn't compress enough)
CONTEXT_DISPLAY_THRESHOLD = 0.3  # Show context usage display threshold
RECENT_MESSAGES_TO_KEEP = 10  # Messages to preserve when summarizing

# Truncation limits (chars)
OUTPUT_TRUNCATION_LIMIT = 8000  # Guards against large outputs filling context
SUMMARY_MSG_LIMIT = 1000  # Guards against large messages filling context when summarizing


# Setup of the environment and some logging. Not neccessary to touch this.
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


PROJECT_ROOT = Path(__file__).resolve().parent
# Only require OpenAI credentials when actually using OpenAI models/embeddings.
if ACTIVE_MODEL_CONFIG.provider == "openai" or EMBEDDING_MODEL_NAME.startswith("openai:"):
    _set_env("OPENAI_API_KEY")

# LangSmith is optional; only enable if provided in environment/.env


logging.getLogger("httpx").setLevel(logging.WARNING)  # Less warnings in the output
logging.getLogger("faiss").setLevel(logging.WARNING)


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


@dataclass(kw_only=True)  # pyright: ignore[reportCallIssue]
class BaseConfiguration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes, including embedding model selection, retriever provider choice, and search parameters.
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
    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default_factory=lambda: EMBEDDING_MODEL_NAME,
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    retriever_provider: Annotated[
        Literal["faiss", "chroma"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="chroma",
        metadata={"description": "The vector store provider to use for retrieval."},
    )

    examples_search_type: Annotated[
        Literal["similarity", "mmr", "similarity_score_threshold"],
        {"__template_metadata__": {"kind": "reranker"}},
    ] = field(
        default="mmr",
        metadata={
            "description": "Defines the type of search that the retriever should perform."
        },
    )

    examples_search_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"k": 2, "fetch_k": 10, "lambda_mult": 0.5},
        metadata={
            "description": "Additional keyword arguments to pass to the search function of the retriever. See langgraph documentation for details about what kwargs works for the different search types. See https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.as_retriever"
        },
    )

    rerank_provider: Annotated[
        Literal["None", "flash"],
        {"__template_metadata__": {"kind": "reranker"}},
    ] = field(
        default="None",
        metadata={
            "description": "The provider user for reranking the retrieved documents."
        },
    )

    rerank_kwargs: dict[str, Any] = field(
        default_factory=lambda: {},
        metadata={"description": "Keyword arguments provided to the reranker"},
    )

    # Models
    agent_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: ACTIVE_MODEL,
        metadata={"description": "The language model used for coding tasks."},
    )
    autonomous_agent_model: Annotated[
        str, {"__template_metadata__": {"kind": "llm"}}
    ] = field(
        default_factory=lambda: ACTIVE_MODEL,
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

    # Logging
    log_to_file: bool = field(
        default=True,
        metadata={
            "description": "Enable writing assistant I/O to a session Markdown log."
        },
    )
    log_dir: str = field(
        default="output/logs",
        metadata={
            "description": "Directory where session logs are stored (auto-created)."
        },
    )
    log_filename_prefix: str = field(
        default="agent_output",
        metadata={"description": "Prefix for per-session log filenames."},
    )
    log_version_info: bool = field(
        default=True,
        metadata={
            "description": "Include Julia/Jutul/JutulDarcy versions in log header."
        },
    )

    # Context management
    show_context_usage: bool = field(
        default=True,
        metadata={
            "description": "Display context usage stats in CLI after each model call."
        },
    )
    context_window_size: int = field(
        default_factory=lambda: ACTIVE_MODEL_CONFIG.context_window,
        metadata={
            "description": "Model context window size in tokens."
        },
    )
    context_summarize_threshold: float = field(
        default=CONTEXT_USAGE_THRESHOLD,
        metadata={
            "description": "Fraction of context at which to start summarizing (0.0-1.0)."
        },
    )
    context_display_threshold: float = field(
        default=CONTEXT_DISPLAY_THRESHOLD,
        metadata={
            "description": "Only show context display when usage exceeds this fraction (0.0-1.0)."
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
        _fields = {f.name for f in fields(cls) if f.init}  # pyright: ignore[reportArgumentType]
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=BaseConfiguration)
