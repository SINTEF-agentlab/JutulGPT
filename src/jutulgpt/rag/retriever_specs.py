from dataclasses import dataclass
from functools import partial
from typing import Callable, Union

from jutulgpt.configuration import PROJECT_ROOT
from jutulgpt.rag import split_docs, split_examples


@dataclass
class RetrieverSpec:
    dir_path: (
        str | Callable[[], str]
    )  # Can be a string or a callable that returns the path
    filetype: Union[str, list[str]]  # Can be a single filetype or a list of filetypes.
    split_func: Callable


def _get_jutuldarcy_docs_path() -> str:
    """Get the path to JutulDarcy docs from the installed Julia package."""
    from jutulgpt.rag.package_path import check_version_and_invalidate, get_package_root

    check_version_and_invalidate()
    root = get_package_root()
    if root is None:
        raise RuntimeError(
            "JutulDarcy is not installed. "
            'Install it in Julia with: using Pkg; Pkg.add("JutulDarcy")'
        )
    return str(root / "docs" / "src" / "man")


def _get_jutuldarcy_examples_path() -> str:
    """Get the path to JutulDarcy examples from the installed Julia package."""
    from jutulgpt.rag.package_path import check_version_and_invalidate, get_package_root

    check_version_and_invalidate()
    root = get_package_root()
    if root is None:
        raise RuntimeError(
            "JutulDarcy is not installed. "
            'Install it in Julia with: using Pkg; Pkg.add("JutulDarcy")'
        )
    return str(root / "examples")


RETRIEVER_SPECS = {
    "jutuldarcy": {
        "docs": RetrieverSpec(
            dir_path=_get_jutuldarcy_docs_path,
            filetype="md",
            split_func=split_docs.split_docs,
        ),
        "examples": RetrieverSpec(
            dir_path=_get_jutuldarcy_examples_path,
            filetype="jl",
            split_func=partial(
                split_examples.split_examples,
                header_to_split_on=1,  # Split on `# #`
            ),
        ),
    },
    "fimbul": {
        "docs": RetrieverSpec(
            dir_path=str(PROJECT_ROOT / "rag" / "fimbul" / "docs" / "man"),
            filetype="md",
            split_func=split_docs.split_docs,
        ),
        "examples": RetrieverSpec(
            dir_path=str(PROJECT_ROOT / "rag" / "fimbul" / "examples"),
            filetype="jl",
            split_func=partial(
                split_examples.split_examples,
                header_to_split_on=1,  # Split on `# #`
            ),
        ),
    },
}
