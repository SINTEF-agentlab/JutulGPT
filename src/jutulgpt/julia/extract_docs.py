"""
Extract function documentation from JutulDarcy using Julia's @doc macro.

This provides actual docstrings rather than just references like in the markdown files.
"""

import json
import pickle
import tomllib
from pathlib import Path
from typing import Optional

from jutulgpt.cli import colorscheme, print_to_console
from jutulgpt.julia.julia_code_runner import run_code_string_direct


def _get_package_version() -> Optional[str]:
    """Read the installed JutulDarcy version from its Project.toml."""
    from jutulgpt.rag.package_path import get_package_root

    root = get_package_root()
    if root is None:
        return None
    project_toml = root / "Project.toml"
    if not project_toml.exists():
        return None
    with open(project_toml, "rb") as f:
        return tomllib.load(f).get("version")


def _default_cache_path() -> Path:
    from jutulgpt.configuration import PROJECT_ROOT

    cache_dir = PROJECT_ROOT.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "jutuldarcy_function_docs.pkl"


def extract_jutuldarcy_documentation(
    cache_path: Optional[Path] = None, force_refresh: bool = False
) -> dict[str, str]:
    """
    Extract all exported function documentation from JutulDarcy.

    This uses Julia's @doc macro to get the actual docstrings for all exported
    functions, types, and constants from JutulDarcy and Jutul.

    Results are cached to disk.  The cache is automatically invalidated when
    the installed JutulDarcy version changes.

    Args:
        cache_path: Path to cache the extracted documentation.
        force_refresh: Force re-extraction even if cache exists.

    Returns:
        Dictionary mapping qualified function names to their documentation strings.
    """
    if cache_path is None:
        cache_path = _default_cache_path()

    current_version = _get_package_version()

    # Check cache first
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            # Support both old format (bare dict) and new format (dict with version key)
            if isinstance(cached, dict) and "version" in cached and "docs" in cached:
                if cached["version"] == current_version:
                    return cached["docs"]
            elif isinstance(cached, dict):
                # Old format without version — treat as stale
                pass
        except Exception as e:
            print_to_console(
                text=f"Failed to load cache: {e}. Re-extracting...",
                title="JutulDarcy Documentation",
                border_style=colorscheme.warning,
            )

    julia_code = """
    using JutulDarcy
    using Jutul
    
    function extract_module_docs(mod::Module)
        docs_dict = Dict{String, String}()
        exported_names = names(mod; all=false, imported=false)
        
        for name in exported_names
            if name in [:eval, :include]
                continue
            end
            
            try
                sym = getfield(mod, name)
                doc = Docs.doc(sym)
                doc_str = string(doc)
                
                if occursin("No documentation found", doc_str) || 
                   (occursin("Binding", doc_str) && length(doc_str) < 100)
                    continue
                end
                
                qualified_name = string(mod, ".", name)
                docs_dict[qualified_name] = doc_str
            catch e
                continue
            end
        end
        
        return docs_dict
    end
    
    all_docs = merge(extract_module_docs(JutulDarcy), extract_module_docs(Jutul))
    
    # Output as JSON
    using JSON
    println(JSON.json(all_docs))
    """

    try:
        stdout, stderr = run_code_string_direct(julia_code)

        if stderr and "ERROR" in stderr:
            print_to_console(
                text=f"Julia extraction error: {stderr}",
                title="JutulDarcy Documentation",
                border_style=colorscheme.error,
            )
            return {}

        # Parse JSON output
        docs = json.loads(stdout.strip())

        # Cache the results with version for future invalidation
        with open(cache_path, "wb") as f:
            pickle.dump({"version": current_version, "docs": docs}, f)

        return docs

    except Exception as e:
        print_to_console(
            text=f"Failed to extract documentation: {e}",
            title="JutulDarcy Documentation",
            border_style=colorscheme.error,
        )
        return {}


def search_function_docs(
    query: str, docs: Optional[dict[str, str]] = None, top_k: int = 5
) -> list[tuple[str, str]]:
    """
    Search for functions matching a query string.

    Args:
        query: Search query (function name or partial name)
        docs: Documentation dictionary (if None, will load from cache)
        top_k: Number of results to return

    Returns:
        List of (function_name, documentation) tuples
    """
    if docs is None:
        docs = extract_jutuldarcy_documentation()

    query_lower = query.lower()
    results = []

    for func_name, doc in docs.items():
        # Simple relevance scoring
        score = 0

        # Exact match in name
        if query_lower in func_name.lower():
            score += 100

        # Match in documentation
        if query_lower in doc.lower():
            score += 10

        if score > 0:
            results.append((func_name, doc, score))

    # Sort by score and return top_k
    results.sort(key=lambda x: x[2], reverse=True)
    return [(name, doc) for name, doc, _ in results[:top_k]]


def format_function_doc(func_name: str, doc: str) -> str:
    """
    Format a function documentation for display.

    Args:
        func_name: Qualified function name
        doc: Documentation string

    Returns:
        Formatted documentation
    """
    return f"# {func_name}\n\n{doc}\n"
