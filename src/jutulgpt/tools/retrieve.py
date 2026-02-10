from __future__ import annotations

import subprocess
from functools import partial
from typing import Annotated, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg, tool
from pydantic import BaseModel, Field

# from jutulgpt import configuration
import jutulgpt.rag.retrieval as retrieval
import jutulgpt.rag.split_examples as split_examples
from jutulgpt.cli import colorscheme, print_to_console
from jutulgpt.configuration import (
    BaseConfiguration,
    cli_mode,
)
from jutulgpt.julia import get_function_documentation_from_list_of_funcs
from jutulgpt.logging import RAGEntry, ToolEntry, get_session_logger
from jutulgpt.rag.package_paths import (
    get_package_docs_path,
    get_package_examples_path,
    get_package_faq_path,
    get_package_root,
)
from jutulgpt.rag.retriever_specs import get_retriever_spec
from jutulgpt.utils.documents import get_file_source


def make_retrieve_tool(
    name: str,
    doc_key: str,
    doc_label: str,
    input_cls: type,
) -> BaseTool:
    @tool(
        name,
        args_schema=input_cls,
        description=f"""Use this tool to look up full examples from the {doc_label} documentation. Use this tool when answering any Julia code question about {doc_label}.""",
    )
    def retrieve_tool(
        query: str, config: Annotated[RunnableConfig, InjectedToolArg]
    ) -> str:
        configuration = BaseConfiguration.from_runnable_config(config)

        # Human interaction: modify query
        if configuration.human_interaction.rag_query:
            if cli_mode:
                from jutulgpt.human_in_the_loop.cli import modify_rag_query

                query = modify_rag_query(query, doc_label)
            else:
                from jutulgpt.human_in_the_loop.ui import modify_rag_query

                query = modify_rag_query(query, doc_label)
        if not query.strip():
            return "The query is empty."

        # Retrieve examples
        with retrieval.make_retriever(
            config=config,
            spec=get_retriever_spec(doc_key, "examples"),
            retrieval_params=retrieval.RetrievalParams(
                search_type=configuration.examples_search_type,
                search_kwargs=configuration.examples_search_kwargs,
            ),
        ) as retriever:
            retrieved_examples = retriever.invoke(query)

        # Format summary of what was retrieved
        if retrieved_examples:
            sources = [get_file_source(doc) for doc in retrieved_examples]
            unique_sources = list(dict.fromkeys(sources))  # Deduplicate while preserving order

            # Make paths relative to package root for cleaner display
            package_root = str(get_package_root(doc_label))
            relative_sources = [
                s.replace(package_root + "/", "") if s.startswith(package_root) else s
                for s in unique_sources
            ]

            summary = f"Retrieved {len(retrieved_examples)} examples:\n" + "\n".join(
                f"- {s}" for s in relative_sources
            )
        else:
            summary = "No examples found"

        # Show results in CLI
        print_to_console(
            text=summary,
            title=f"Retrieving from {doc_label} examples",
            border_style=colorscheme.message,
        )

        # Log RAG retrieval with results
        logger = get_session_logger()
        if logger:
            logger.log(
                RAGEntry(
                    content=summary,
                    title=f"Retrieving from {doc_label} examples",
                    query=query,
                    source=doc_label,
                    num_results=len(retrieved_examples) if retrieved_examples else 0,
                )
            )

        # Human interaction: filter docs/examples
        if configuration.human_interaction.retrieved_examples:
            if cli_mode:
                from jutulgpt.human_in_the_loop.cli import response_on_rag

                if configuration.human_interaction.retrieved_examples:
                    retrieved_examples = response_on_rag(
                        docs=retrieved_examples,
                        get_file_source=get_file_source,
                        get_section_path=split_examples.get_section_path,
                        format_doc=partial(
                            split_examples.format_doc, within_julia_context=False
                        ),
                        action_name=f"Modify retrieved {doc_label} examples",
                        edit_julia_file=True,
                    )
            else:
                from jutulgpt.human_in_the_loop.ui import response_on_rag

                if configuration.human_interaction.retrieved_examples:
                    retrieved_examples = response_on_rag(
                        retrieved_examples,
                        get_file_source=get_file_source,
                        get_section_path=split_examples.get_section_path,
                        format_doc=split_examples.format_doc,
                        action_name=f"Modify retrieved {doc_label} examples",
                    )

        examples = split_examples.format_examples(retrieved_examples)

        format_str = lambda s: s if s != "" else "(empty)"
        out = format_str(examples)
        return out

    return retrieve_tool


# Input schemas
class RetrieveJutulDarcyInput(BaseModel):
    query: str = Field(
        "The query that will be used for document and example retrieval",
    )

# Create tools
retrieve_jutuldarcy_examples = make_retrieve_tool(
    name="retrieve_jutuldarcy_examples",
    doc_key="jutuldarcy",
    doc_label="JutulDarcy",
    input_cls=RetrieveJutulDarcyInput,
)

class RetrieveFunctionDocumentationInput(BaseModel):
    function_names: List[str] = Field(
        description="A list of function names to retrieve the documentation for.",
    )


@tool(
    "retrieve_function_documentation",
    description="Retrieve documentation for specific Julia functions. Use this tool when needing detailed information about function signatures and usage.",
    args_schema=RetrieveFunctionDocumentationInput,
)
def retrieve_function_documentation(
    function_names: List[str],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    _, retrieved_signatures = get_function_documentation_from_list_of_funcs(
        func_names=function_names
    )

    # Log the retrieval
    logger = get_session_logger()
    if logger:
        logger.log(
            ToolEntry(
                content=retrieved_signatures or "No documentation found",
                title="Function Documentation Retriever",
                tool_name="retrieve_function_documentation",
                args={"function_names": function_names},
            )
        )

    if retrieved_signatures:
        return retrieved_signatures

    return "No function signatures found for the provided function names."


class GrepSearchInput(BaseModel):
    """Input for grep search tool."""

    query: str = Field(
        description=(
            "The search pattern. With isRegexp=false (default), searches for the exact phrase. "
            "With isRegexp=true, supports regex patterns (e.g. 'CartesianMesh|setup_well' for OR). "
            "For multiple unrelated terms, prefer separate searches."
        )
    )
    includePattern: Optional[str] = Field(
        default=None,
        description=(
            "Optional file pattern to restrict search (e.g. '*.jl' or '*.md'). "
            "If omitted, searches both '*.jl' and '*.md' (recommended for discovery). "
            "Use '*.md' for conceptual/how-to documentation and '*.jl' for code/examples/API usage. "
            "Note: Use simple patterns like '*.jl' or '*.md', not glob patterns like '**/*.jl' and '**/*.md' - "
            "the search is already recursive."
        ),
    )
    isRegexp: Optional[bool] = Field(
        default=False,
        description=(
            "If false (default), query is treated as exact phrase/substring. "
            "If true, query is a regex pattern supporting OR (|), wildcards, etc."
        ),
    )


@tool(
    "grep_search",
    description=(
        "Search for keywords in JutulDarcy documentation and examples. "
        "Searches recursively through all files. Limited to first 20 matches. "
        "Returns file paths and line numbers with matching content. "
        "Use this to discover which files contain relevant code before reading them with the file-reader tool. "
        "For broad discovery, leave includePattern unset; only set it when you want one file type."
    ),
    args_schema=GrepSearchInput,
)
def grep_search(
    query: str,
    includePattern: Optional[str] = None,
    isRegexp: Optional[bool] = False,
) -> str:
    try:
        package_root = get_package_root("JutulDarcy")
        search_paths = [
            get_package_examples_path("JutulDarcy"),
            get_package_docs_path("JutulDarcy"),
            get_package_faq_path("JutulDarcy"),
        ]
        if not search_paths:
            raise FileNotFoundError(
                "No searchable JutulDarcy examples/docs/faq paths were found."
            )

        cmd_parts = ["grep", "-r", "-n", "-i"]

        if isRegexp:
            cmd_parts.append("-E")
        else:
            cmd_parts.append("-F")  # Fixed string search

        if includePattern:
            cmd_parts.extend(["--include", includePattern])
        else:
            cmd_parts.extend(
                [
                    "--include=*.jl",
                    "--include=*.md",
                ]
            )

        cmd_parts.append(query)
        cmd_parts.extend(search_paths)

        result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=10)
        if result.stdout:
            lines = result.stdout.strip().split("\n")
            file_counts: dict[str, int] = {}

            # Count matches per file
            for line in lines:
                parts = line.split(":", 2)
                if len(parts) == 3:
                    file_counts[parts[0]] = file_counts.get(parts[0], 0) + 1

            # Sort files by match count (descending)
            sorted_files = sorted(file_counts.items(), key=lambda x: (-x[1], x[0]))

            # Format file list with relative paths
            file_list = []
            for filepath, count in sorted_files:
                rel_path = filepath.replace(str(package_root) + "/", "")
                file_list.append(f"- {rel_path} ({count} {'match' if count == 1 else 'matches'})")

            # Build match results for agent (limit to 20)
            match_results = []
            for line in lines[:20]:
                parts = line.split(":", 2)
                if len(parts) == 3:
                    match_results.append(f"File: {parts[0]}, Line {parts[1]}: {parts[2]}")
                else:
                    match_results.append(line)

            # Terminal output
            total_matches = len(lines)
            num_files = len(file_counts)
            if len(file_list) > 3:
                terminal_output = f"Top 3 of {num_files} files ({total_matches} matches total):\n"
                terminal_output += "\n".join(file_list[:3])
                terminal_output += f"\n+ {num_files - 3} more files"
            else:
                terminal_output = f"{total_matches} matches in {num_files} files:\n" + "\n".join(file_list)

            print_to_console(
                text=terminal_output,
                title=f"Grep search: {query}",
                border_style=colorscheme.message,
            )

            # Log: Show all files
            log_output = f"Found {total_matches} matches in {num_files} files:\n" + "\n".join(file_list)

            logger = get_session_logger()
            if logger:
                logger.log(
                    ToolEntry(
                        content=log_output,
                        title=f"Grep search: {query}",
                        tool_name="grep_search",
                        args={"query": query, "includePattern": includePattern},
                    )
                )

            # Return full details to agent
            return f"Found {total_matches} matches:\n" + "\n\n".join(match_results)
        else:
            no_match_msg = f"No matches found for: {query}"
            print_to_console(
                text=no_match_msg,
                title=f"Grep search: {query}",
                border_style=colorscheme.message,
            )
            logger = get_session_logger()
            if logger:
                logger.log(
                    ToolEntry(
                        content=no_match_msg,
                        title=f"Grep search: {query}",
                        tool_name="grep_search",
                        args={"query": query, "includePattern": includePattern},
                    )
                )
            return no_match_msg

    except Exception as e:
        error_msg = f"Error during text search: {str(e)}"
        print_to_console(
            text=error_msg,
            title=f"Grep search error: {query}",
            border_style=colorscheme.error,
        )
        logger = get_session_logger()
        if logger:
            logger.log(
                ToolEntry(
                    content=error_msg,
                    title=f"Grep search error: {query}",
                    tool_name="grep_search",
                    args={"query": query, "includePattern": includePattern},
                    error=str(e),
                )
            )
        return error_msg
