"""
Extract function documentation from JutulDarcy using Julia's @doc macro.

This provides actual docstrings rather than just references like in the markdown files.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

from jutulgpt.cli import colorscheme, print_to_console
from jutulgpt.julia.julia_code_runner import run_code_string_direct


def extract_jutuldarcy_documentation(
    cache_path: Optional[Path] = None, force_refresh: bool = False
) -> dict[str, str]:
    """
    Extract all exported function documentation from JutulDarcy.

    This uses Julia's @doc macro to get the actual docstrings for all exported
    functions, types, and constants from JutulDarcy and Jutul.

    Args:
        cache_path: Path to cache the extracted documentation (default: PROJECT_ROOT / ".cache" / "jutuldarcy_docs.pkl")
        force_refresh: Force re-extraction even if cache exists

    Returns:
        Dictionary mapping qualified function names to their documentation strings
    """
    if cache_path is None:
        from jutulgpt.configuration import PROJECT_ROOT

        cache_dir = PROJECT_ROOT.parent / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "jutuldarcy_function_docs.pkl"

    # Check cache first
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                docs = pickle.load(f)
                # Silently load from cache - no need to announce this every time
                return docs
        except Exception as e:
            print_to_console(
                text=f"Failed to load cache: {e}. Re-extracting...",
                title="JutulDarcy Documentation",
                border_style=colorscheme.warning,
            )

    # Extract documentation using Julia
    # print_to_console(
    #     text="Extracting function documentation from JutulDarcy (this may take a moment)...",
    #     title="JutulDarcy Documentation",
    #     border_style=colorscheme.message,
    # )

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

        # Cache the results
        with open(cache_path, "wb") as f:
            pickle.dump(docs, f)

        # print_to_console(
        #     text=f"Extracted {len(docs)} function documentations",
        #     title="JutulDarcy Documentation",
        #     border_style=colorscheme.success,
        # )

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
