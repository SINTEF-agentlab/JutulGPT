"""Retriever specifications for different document sets.

JutulDarcy specs resolve paths lazily from the installed Julia package so they
always stay in sync with the version the user has installed.
"""

import hashlib
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


def _persist_path(name: str, provider: str, suffix: str) -> str:
    from jutulgpt.configuration import PROJECT_ROOT

    path = (
        PROJECT_ROOT
        / "rag"
        / "retriever_store"
        / f"retriever_{name}_{provider}_{suffix}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _cache_path(name: str, suffix: str) -> str:
    from jutulgpt.configuration import PROJECT_ROOT

    path = PROJECT_ROOT / "rag" / "loaded_store" / f"loaded_{name}_{suffix}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _suffix_from_path(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=None)
def get_retriever_spec(package: str, doc_type: str) -> RetrieverSpec:
    """Return a :class:`RetrieverSpec`, resolving package paths lazily.

    The first call for a given *(package, doc_type)* pair triggers a Julia
    subprocess to locate the installed package; subsequent calls return the
    cached result instantly.
    """
    if package == "jutuldarcy":
        return _get_jutuldarcy_spec(doc_type)
    raise ValueError(f"Unknown package: {package}")


# ---------------------------------------------------------------------------
# JutulDarcy – paths resolved from the installed Julia package
# ---------------------------------------------------------------------------


def _get_jutuldarcy_spec(doc_type: str) -> RetrieverSpec:
    from jutulgpt.rag.package_paths import (
        get_package_docs_path,
        get_package_examples_path,
        get_package_root,
    )

    package_suffix = _suffix_from_path(str(get_package_root("JutulDarcy")))

    if doc_type == "docs":
        return RetrieverSpec(
            dir_path=get_package_docs_path("JutulDarcy"),
            filetype="md",
            split_func=split_docs.split_docs,
            persist_path=lambda provider: _persist_path(
                "jutuldarcy_docs", provider, package_suffix
            ),
            cache_path=_cache_path("jutuldarcy_docs", package_suffix),
            collection_name=f"jutuldarcy_docs_{package_suffix}",
        )
    elif doc_type == "examples":
        return RetrieverSpec(
            dir_path=get_package_examples_path("JutulDarcy"),
            filetype="jl",
            split_func=partial(
                split_examples.split_examples,
                header_to_split_on=1,  # Split on `# #`
            ),
            persist_path=lambda provider: _persist_path(
                "jutuldarcy_examples", provider, package_suffix
            ),
            cache_path=_cache_path("jutuldarcy_examples", package_suffix),
            collection_name=f"jutuldarcy_examples_{package_suffix}",
        )
    raise ValueError(f"Unknown doc_type for jutuldarcy: {doc_type}")
