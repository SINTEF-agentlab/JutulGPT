"""Load user configuration from ``jutulgpt.toml``.

Search order:
1. Current working directory
2. Project root (two levels up from this file, i.e. the repo root)

If the file is missing the loader returns an empty dict so that all
downstream code falls back to its Python defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

_CONFIG_FILENAME = "jutulgpt.toml"


def _find_config_file() -> Path | None:
    """Return the first ``jutulgpt.toml`` found, or *None*."""
    # 1. CWD
    cwd_path = Path.cwd() / _CONFIG_FILENAME
    if cwd_path.is_file():
        return cwd_path
    # 2. Project / repo root (../../ relative to this file)
    pkg_path = Path(__file__).resolve().parent.parent.parent / _CONFIG_FILENAME
    if pkg_path.is_file():
        return pkg_path
    return None


def load_config() -> dict:
    """Read and return the TOML config as a nested dict.

    Returns an empty dict when no config file is found.
    """
    path = _find_config_file()
    if path is None:
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def get(config: dict, *keys, default=None):
    """Safe nested dict access.

    >>> cfg = {"model": {"preset": "gpt-4.1"}}
    >>> get(cfg, "model", "preset", default="gpt-5.2-reasoning")
    'gpt-4.1'
    >>> get(cfg, "missing", "key", default=42)
    42
    """
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
