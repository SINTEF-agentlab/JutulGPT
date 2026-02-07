from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from jutulgpt import configuration as cfg


@dataclass(frozen=True)
class CliArgs:
    model: str


def _normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def resolve_model_config(name: str) -> cfg.ModelConfig:
    """Resolve a CLI `--model` string to a `ModelConfig` preset."""
    key = _normalize_model_name(name)

    presets: dict[str, cfg.ModelConfig] = {
        # OpenAI
        "gpt-4.1": cfg.OPENAI_GPT_4_1,
        "gpt-5-mini": cfg.OPENAI_GPT_5_MINI,
        "gpt-5-mini-reasoning": cfg.OPENAI_GPT_5_MINI_REASONING,
        "gpt-5.1": cfg.OPENAI_GPT_5_1,
        "gpt-5.1-reasoning": cfg.OPENAI_GPT_5_1_REASONING,
        "gpt-5.2": cfg.OPENAI_GPT_5_2,
        "gpt-5.2-reasoning": cfg.OPENAI_GPT_5_2_REASONING,
        # Ollama / Qwen
        "qwen3:14b": cfg.OLLAMA_QWEN3_14B,
        "qwen3:14b-thinking": cfg.OLLAMA_QWEN3_14B_THINKING,
    }

    if key in presets:
        return presets[key]

    # Also allow passing a raw fully-qualified model like "openai:gpt-4.1"
    # (will fall back to minimal defaults elsewhere).
    if ":" in name:
        provider, model = name.split(":", maxsplit=1)
        provider = provider.strip().lower()
        if provider in ("openai", "ollama") and model.strip():
            return cfg.ModelConfig(provider=provider, model=model.strip())

    raise ValueError(
        f"Unknown model '{name}'. Supported: {', '.join(sorted(presets.keys()))}"
    )


def parse_cli_args(argv: Optional[list[str]] = None) -> CliArgs:
    """Parse CLI args for JutulGPT standalone runs."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--model",
        default=cfg.DEFAULT_MODEL_PRESET,
        help=(
            "Model preset to use. Examples: "
            "gpt-4.1, gpt-5.2, gpt-5.2-reasoning, "
            "qwen3:14b, qwen3:14b-thinking"
        ),
    )
    ns = parser.parse_args(argv)
    return CliArgs(model=ns.model)


def apply_model_from_cli(model: str) -> None:
    """Apply the CLI model preset globally for this process."""
    cfg.ACTIVE_MODEL_CONFIG = resolve_model_config(model)  # type: ignore[misc]
    # Update derived globals in configuration module
    cfg.ACTIVE_PROVIDER = cfg.ACTIVE_MODEL_CONFIG.provider
    cfg.ACTIVE_MODEL_NAME = cfg.ACTIVE_MODEL_CONFIG.model
    cfg.ACTIVE_MODEL = f"{cfg.ACTIVE_PROVIDER}:{cfg.ACTIVE_MODEL_NAME}"
    cfg.MODEL_CONTEXT_WINDOW = cfg.ACTIVE_MODEL_CONFIG.context_window
    cfg.EMBEDDING_MODEL_NAME = cfg._EMBEDDING_MODEL_BY_PROVIDER[cfg.ACTIVE_PROVIDER]

