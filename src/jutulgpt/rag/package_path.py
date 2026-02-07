"""
Resolve JutulDarcy's installed package path from Julia.

The package root is resolved via ``pathof(JutulDarcy)`` and cached in-memory
for the lifetime of the process.  Set ``[retrieval].package_path`` in
``jutulgpt.toml`` to skip Julia startup entirely.
"""

from pathlib import Path
from typing import Optional

from jutulgpt.cli import colorscheme, print_to_console

_cached_root: Optional[Path] = None


def get_package_root() -> Optional[Path]:
    """Return the root directory of the installed JutulDarcy Julia package.

    Resolution order:
    1. ``jutulgpt.toml`` ``[retrieval].package_path`` (explicit, no Julia needed).
    2. In-memory cache (per-process).
    3. ``julia -e 'using JutulDarcy; println(pathof(JutulDarcy))'`` (slow, once per process).

    Returns ``None`` if JutulDarcy is not installed.
    """
    global _cached_root

    # 1. TOML config (explicit path, skips Julia call entirely)
    from jutulgpt.configuration import _toml

    toml_path = _toml.get("retrieval", {}).get("package_path")
    if toml_path:
        p = Path(toml_path)
        if p.is_dir():
            _cached_root = p
            return _cached_root

    # 2. In-memory cache
    if _cached_root is not None:
        return _cached_root

    # 3. Julia call (result cached in memory for the rest of the process)
    root = _resolve_from_julia()
    if root is not None:
        _cached_root = root
    return root


def _resolve_from_julia() -> Optional[Path]:
    """Call Julia to find the package root via ``pathof(JutulDarcy)``."""
    from jutulgpt.julia.julia_code_runner import run_code_string_direct

    code = "using JutulDarcy; println(pathof(JutulDarcy))"
    try:
        result, _ = run_code_string_direct(code)
        src_file = Path(result.strip())  # e.g. .../JutulDarcy/src/JutulDarcy.jl
        root = src_file.parent.parent  # .../JutulDarcy/
        if root.is_dir():
            print_to_console(
                text=f"Resolved JutulDarcy at {root}",
                title="Package Path",
                border_style=colorscheme.success,
            )
            return root
    except Exception as e:
        print_to_console(
            text=f"Could not resolve JutulDarcy path: {e}",
            title="Package Path",
            border_style=colorscheme.warning,
        )
    return None
