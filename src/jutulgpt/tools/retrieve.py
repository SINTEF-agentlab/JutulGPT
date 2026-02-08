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
from jutulgpt.rag.package_paths import get_package_root
from jutulgpt.rag.retriever_specs import get_retriever_spec
from jutulgpt.utils import get_file_source


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
            summary = f"Retrieved {len(retrieved_examples)} examples:\n" + "\n".join(f"- {s}" for s in sources)
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


class RetrieveFimbulInput(BaseModel):
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

retrieve_fimbul = make_retrieve_tool(
    name="retrieve_fimbul",
    doc_key="fimbul",
    doc_label="Fimbul",
    input_cls=RetrieveFimbulInput,
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
        description="The keyword based pattern to search for in files. Can be a regex or plain text pattern"
    )
    includePattern: Optional[str] = Field(
        default=None,
        description=(
            "File pattern to search (e.g. '*.jl' or '*.md'). "
            "Defaults to '*.jl' and '*.md'. "
            "Note: Use simple patterns like '*.jl', not glob patterns like '**/*.jl' - "
            "the search is already recursive."
        ),
    )
    isRegexp: Optional[bool] = Field(
        default=False, description="Whether the pattern is a regex."
    )


@tool(
    "grep_search",
    description=(
        "Search for keywords in JutulDarcy documentation and examples. "
        "Searches recursively through all files. Limited to first 20 matches. "
        "Returns file paths and line numbers with matching content. "
        "Use this to discover which files contain relevant code before reading them with the file-reader tool."
    ),
    args_schema=GrepSearchInput,
)
def grep_search(
    query: str,
    includePattern: Optional[str] = None,
    isRegexp: Optional[bool] = False,
) -> str:
    try:
        workspace_path = str(get_package_root("JutulDarcy"))
        cmd_parts = ["grep", "-r", "-n"]

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

        cmd_parts.extend([query, workspace_path])

        result = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=10)

        if result.stdout:
            lines = result.stdout.strip().split("\n")[:20]  # Limit to 20 results
            match_results = []  # Full details for agent
            file_counts: dict[str, int] = {}
            for match in lines:
                # Parse: filename:line_number:content
                parts = match.split(":", 2)
                if len(parts) != 3:
                    match_results.append(match)
                    continue
                filename, line_str, content = parts
                match_results.append(f"File: {filename}, Line {line_str}: {content}")
                file_counts[filename] = file_counts.get(filename, 0) + 1

            # Build file list with relative paths
            file_list = []
            for filepath, count in file_counts.items():
                rel_path = filepath.replace(str(workspace_path) + "/", "")
                match_text = "match" if count == 1 else "matches"
                file_list.append(f"- {rel_path} ({count} {match_text})")

            match_word = "match" if len(match_results) == 1 else "matches"
            file_word = "file" if len(file_counts) == 1 else "files"

            # Terminal: Show first 3 files only
            if len(file_list) > 3:
                terminal_output = f"First 3 of {len(file_counts)} {file_word} ({len(match_results)} {match_word} total):\n"
                terminal_output += "\n".join(file_list[:3])
                terminal_output += f"\n+ {len(file_list) - 3} more {file_word}"
            else:
                terminal_output = f"{len(match_results)} {match_word} in {len(file_counts)} {file_word}:\n" + "\n".join(file_list)

            print_to_console(
                text=terminal_output,
                title=f"Grep search: {query}",
                border_style=colorscheme.message,
            )

            # Log: Show all files
            log_output = f"Found {len(match_results)} {match_word} in {len(file_counts)} {file_word}:\n" + "\n".join(file_list)

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
            return f"Found {len(match_results)} {match_word}:\n" + "\n\n".join(match_results)
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
