"""Retriever specifications for different document sets.

JutulDarcy specs resolve paths lazily from the installed Julia package so they
always stay in sync with the version the user has installed.
"""

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Callable, Optional, Union

from jutulgpt.rag import split_docs, split_examples


@dataclass
class RetrieverSpec:
    dir_path: str
    filetype: Union[str, list[str]]
    split_func: Callable
    # Only needed for vector-store retrievers (not BM25):
    persist_path: Optional[Callable] = None
    cache_path: Optional[str] = None
    collection_name: Optional[str] = None


@lru_cache(maxsize=None)
def get_retriever_spec(package: str, doc_type: str) -> RetrieverSpec:
    """Return a :class:`RetrieverSpec`, resolving package paths lazily.

    The first call for a given *(package, doc_type)* pair triggers a Julia
    subprocess to locate the installed package; subsequent calls return the
    cached result instantly.
    """
    if package == "jutuldarcy":
        return _get_jutuldarcy_spec(doc_type)
    elif package == "fimbul":
        return _get_fimbul_spec(doc_type)
    raise ValueError(f"Unknown package: {package}")


# ---------------------------------------------------------------------------
# JutulDarcy – paths resolved from the installed Julia package
# ---------------------------------------------------------------------------

def _get_jutuldarcy_spec(doc_type: str) -> RetrieverSpec:
    from jutulgpt.rag.package_paths import get_package_docs_path, get_package_examples_path

    if doc_type == "docs":
        return RetrieverSpec(
            dir_path=get_package_docs_path("JutulDarcy"),
            filetype="md",
            split_func=split_docs.split_docs,
        )
    elif doc_type == "examples":
        return RetrieverSpec(
            dir_path=get_package_examples_path("JutulDarcy"),
            filetype="jl",
            split_func=partial(
                split_examples.split_examples,
                header_to_split_on=1,  # Split on `# #`
            ),
        )
    raise ValueError(f"Unknown doc_type for jutuldarcy: {doc_type}")


# ---------------------------------------------------------------------------
# Fimbul – still uses local rag/ copies
# ---------------------------------------------------------------------------

def _get_fimbul_spec(doc_type: str) -> RetrieverSpec:
    from jutulgpt.configuration import PROJECT_ROOT

    if doc_type == "docs":
        return RetrieverSpec(
            dir_path=str(PROJECT_ROOT / "rag" / "fimbul" / "docs" / "man"),
            filetype="md",
            split_func=split_docs.split_docs,
        )
    elif doc_type == "examples":
        return RetrieverSpec(
            dir_path=str(PROJECT_ROOT / "rag" / "fimbul" / "examples"),
            filetype="jl",
            split_func=partial(
                split_examples.split_examples,
                header_to_split_on=1,  # Split on `# #`
            ),
        )
    raise ValueError(f"Unknown doc_type for fimbul: {doc_type}")
