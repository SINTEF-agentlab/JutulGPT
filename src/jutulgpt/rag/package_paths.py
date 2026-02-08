"""Resolve paths to installed Julia packages for document retrieval.

Instead of bundling copies of JutulDarcy docs/examples locally, we resolve the
package root from the Julia installation via ``pathof()`` and read directly from
there.  Results are cached for the lifetime of the process so the Julia
subprocess is only spawned once per package.
"""

import logging
import os
import subprocess
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def get_package_root(package_name: str) -> Path:
    """Return the root directory of an installed Julia package.

    Runs ``julia --project=<cwd> -e 'using Pkg; …'`` (with the real package
    name) and caches the result so subsequent calls are free.
    """
    julia_code = (
        f"using {package_name}; "
        f"print(dirname(dirname(pathof({package_name}))))"
    )
    project_dir = os.getcwd()
    try:
        result = subprocess.run(
            ["julia", f"--project={project_dir}", "--startup-file=no", "-e", julia_code],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0 and result.stdout.strip():
            path = Path(result.stdout.strip())
            if path.exists():
                logger.info("Resolved %s package root: %s", package_name, path)
                return path
            raise FileNotFoundError(
                f"Resolved path does not exist: {path}"
            )
        raise RuntimeError(
            f"Julia failed to resolve {package_name} "
            f"(rc={result.returncode}): {result.stderr.strip()}"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Timeout while resolving {package_name} package path. "
            "Is Julia installed and is JutulDarcy available?"
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Julia executable not found. "
            "Ensure Julia is installed and available on PATH."
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
    raise FileNotFoundError(
        f"{package_name} documentation not found. Tried: {tried}"
    )


def get_package_examples_path(package_name: str, subdir: str = "examples") -> str:
    """Return the examples directory for a Julia package."""
    root = get_package_root(package_name)
    examples_path = root / subdir
    if not examples_path.exists():
        raise FileNotFoundError(
            f"{package_name} examples not found at {examples_path}"
        )
    return str(examples_path)
