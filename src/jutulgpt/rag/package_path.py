"""
Resolve JutulDarcy's installed package path from Julia and manage cache invalidation.

The package root is resolved via ``pathof(JutulDarcy)`` and cached both in-memory and on disk so that subsequent runs avoid Julia startup.
"""

import json
import shutil
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
    1. In-memory cache (fastest, per-process).
    2. ``metadata.json`` on disk (persists across runs).
    3. ``julia -e 'using JutulDarcy; println(pathof(JutulDarcy))'`` (slow).

    Returns ``None`` if JutulDarcy is not installed.
    """
    global _cached_root

    # 1. In-memory
    if _cached_root is not None:
        return _cached_root

    # 2. Disk metadata
    meta = _load_metadata()
    disk_path = meta.get("package_path")
    if disk_path:
        p = Path(disk_path)
        if p.is_dir():
            _cached_root = p
            return _cached_root

    # 3. Julia call
    root = _resolve_from_julia()
    if root is not None:
        _cached_root = root
        meta["package_path"] = str(root)
        _save_metadata(meta)
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

    If the version changed (or is being recorded for the first time), all
    derived caches (vector stores, loaded docs pickles, function docs) are
    deleted and the metadata is updated.

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
    """Delete vector stores, loaded-doc pickles, and function-docs cache."""
    from jutulgpt.configuration import PROJECT_ROOT

    cache_root = PROJECT_ROOT.parent / ".cache"
    cleared = 0

    # Loaded docs pickles
    loaded_store = cache_root / "loaded_store"
    if loaded_store.exists():
        for p in loaded_store.glob("loaded_jutuldarcy_*.pkl"):
            p.unlink()
            cleared += 1

    # Retriever vector stores
    retriever_store = cache_root / "retriever_store"
    if retriever_store.exists():
        for p in retriever_store.glob("retriever_jutuldarcy_*"):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            cleared += 1

    # Function docs pickle
    func_docs = cache_root / "jutuldarcy_function_docs.pkl"
    if func_docs.exists():
        func_docs.unlink()
        cleared += 1

    # Clean up old GitHub-fetched directories
    jd_cache = cache_root / "jutuldarcy"
    for subdir in ("docs", "examples"):
        old = jd_cache / subdir
        if old.exists():
            shutil.rmtree(old)
            cleared += 1

    if cleared:
        print_to_console(
            text=f"Cleared {cleared} derived cache(s)",
            title="Package Path",
            border_style=colorscheme.success,
        )
