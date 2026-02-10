"""Resolve paths to installed Julia packages for document retrieval.

Instead of bundling copies of JutulDarcy docs/examples locally, we resolve the
package root from the Julia installation via ``pathof()`` and read directly from
there.  Results are cached for the lifetime of the process so the Julia
subprocess is only spawned once per package.
"""

import logging
from functools import lru_cache
from pathlib import Path

from jutulgpt.julia.julia_code_runner import run_code_string_direct

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_package_root(package_name: str) -> Path:
    """Return the root directory of an installed Julia package.

    Uses :func:`run_code_string_direct` to run a small Julia snippet that
    prints the package root.  The result is cached so the Julia subprocess is
    only spawned once per package per session.
    """
    julia_code = f"using {package_name}; println(dirname(dirname(pathof({package_name}))))"
    stdout, stderr = run_code_string_direct(
        julia_code,
        startup_file=False,
        history_file=False,
    )

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) == 1:
        path = Path(lines[0]).expanduser()
        if path.exists():
            logger.info("Resolved %s package root: %s", package_name, path)
            return path
        raise FileNotFoundError(f"Resolved path does not exist: {path}")

    if lines:
        raise RuntimeError(
            "Unexpected Julia output while resolving package path. "
            f"Expected exactly one path line, got {len(lines)} lines: {lines}"
        )

    raise RuntimeError(
        f"Failed to resolve {package_name} package path. Julia stderr: {stderr.strip()}"
    )


def get_package_docs_path(package_name: str, subdirs: list[str] | None = None) -> str:
    """Return the documentation directory for a Julia package.

    Tries each candidate *subdir* (relative to the package root) in order and
    returns the first one that exists.
    """
    if subdirs is None:
        subdirs = ["docs/src/man", "docs/man"]

    root = get_package_root(package_name)
    for subdir in subdirs:
        candidate = root / subdir
        if candidate.exists():
            return str(candidate)

    tried = ", ".join(str(root / s) for s in subdirs)
    raise FileNotFoundError(f"{package_name} documentation not found. Tried: {tried}")


def get_package_examples_path(package_name: str, subdir: str = "examples") -> str:
    """Return the examples directory for a Julia package."""
    root = get_package_root(package_name)
    examples_path = root / subdir
    if not examples_path.exists():
        raise FileNotFoundError(f"{package_name} examples not found at {examples_path}")
    return str(examples_path)
