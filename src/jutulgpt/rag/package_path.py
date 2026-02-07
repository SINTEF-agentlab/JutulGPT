"""
Resolve JutulDarcy's installed package path from Julia and manage cache invalidation.

The package root is resolved via ``pathof(JutulDarcy)`` and cached in-memory
for the lifetime of the process.  Set ``[retrieval].package_path`` in
``jutulgpt.toml`` to skip Julia startup entirely.
"""

import json
import sys
from pathlib import Path
from typing import Optional

from jutulgpt.cli import colorscheme, print_to_console

_cached_root: Optional[Path] = None
_version_checked: bool = False


def _cache_dir() -> Path:
    from jutulgpt.configuration import PROJECT_ROOT

    d = PROJECT_ROOT.parent / ".cache" / "jutuldarcy"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _metadata_path() -> Path:
    return _cache_dir() / "metadata.json"


def _load_metadata() -> dict:
    p = _metadata_path()
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}


def _save_metadata(data: dict) -> None:
    with open(_metadata_path(), "w") as f:
        json.dump(data, f, indent=2)


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


def get_package_version(root: Path) -> Optional[str]:
    """Read the version string from ``Project.toml`` in *root*."""
    project_toml = root / "Project.toml"
    if not project_toml.exists():
        return None

    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomllib  # type: ignore[import]
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[import,no-redef]

    with open(project_toml, "rb") as f:
        data = tomllib.load(f)
    return data.get("version")


def check_version_and_invalidate() -> bool:
    """Compare the current package version to the cached one.

    If the version changed (or is being recorded for the first time), the
    function docs cache is deleted and the metadata is updated.

    Returns ``True`` if caches were invalidated.
    """
    global _version_checked

    if _version_checked:
        return False

    _version_checked = True

    root = get_package_root()
    if root is None:
        return False

    current_version = get_package_version(root)
    meta = _load_metadata()
    cached_version = meta.get("package_version")

    if current_version and current_version != cached_version:
        print_to_console(
            text=f"JutulDarcy version changed: {cached_version} -> {current_version}",
            title="Package Path",
            border_style=colorscheme.message,
        )
        _clear_derived_caches()
        meta["package_version"] = current_version
        _save_metadata(meta)
        return True

    return False


def _clear_derived_caches() -> None:
    """Delete the function-docs cache when the package version changes."""
    from jutulgpt.configuration import PROJECT_ROOT

    cache_root = PROJECT_ROOT.parent / ".cache"

    # Function docs pickle (Julia @doc extraction is slow, so it's cached)
    func_docs = cache_root / "jutuldarcy_function_docs.pkl"
    if func_docs.exists():
        func_docs.unlink()
        print_to_console(
            text="Cleared function docs cache",
            title="Package Path",
            border_style=colorscheme.success,
        )
