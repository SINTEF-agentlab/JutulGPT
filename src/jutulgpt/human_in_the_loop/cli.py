from typing import Callable, List

from langchain_core.documents import Document
from rich.prompt import Prompt
from rich.table import Table

import jutulgpt.cli.cli_utils as utils
from jutulgpt.cli.cli_colorscheme import colorscheme
from jutulgpt.globals import console
from jutulgpt.logging.session import get_session_logger
from jutulgpt.rag.utils import modify_doc_content
from jutulgpt.utils.code_parsing import add_julia_context

from .interactions import (
    Action,
    CHECK_CODE,
    ON_ERROR,
    RAG_DOCS,
    RAG_QUERY,
    TERMINAL_RUN,
    Interaction,
    Option,
)


def _prompt(interaction: Interaction) -> Option:
    """Display interaction options and get user choice.

    Args:
        interaction: The interaction definition

    Returns:
        Option: The selected option
    """
    console.print(f"\n[bold yellow]{interaction.title}[/bold yellow]")

    keys = []
    for i, opt in enumerate(interaction.options, 1):
        label = f"{i}. {opt.label}"
        console.print(label)
        keys.append(str(i))

    default_idx = str(
        next(
            i
            for i, o in enumerate(interaction.options, 1)
            if o.action == interaction.default
        )
    )
    choice = Prompt.ask("Your choice", choices=keys, default=default_idx)

    selected_option = interaction.options[int(choice) - 1]

    # Log the interaction
    logger = get_session_logger()
    if logger:
        logger.log_interaction(
            interaction, selected_option.action, selected_option.label
        )

    return selected_option


def response_on_rag(
    docs: List[Document],
    get_file_source: Callable,
    get_section_path: Callable,
    format_doc: Callable,
    action_name: str = "Modify retrieved documents",
    edit_julia_file: bool = False,
) -> List[Document]:
    """CLI version of response_on_rag that allows interactive document filtering/editing.

    Args:
        docs: List of retrieved documents
        get_file_source: Function to get the file source of a document
        get_section_path: Function to get the section path of a document
        format_doc: Function to format a document for display
        action_name: Name of the action for display
        edit_julia_file: Whether to wrap content in Julia code blocks

    Returns:
        List of documents after user interaction
    """
    if not docs:
        console.print("[yellow]No documents retrieved.[/yellow]")
        return docs

    console.print(f"\n[bold blue]{action_name}[/bold blue]")
    console.print(f"Found {len(docs)} document(s). Choose what to do:")

    selected_option = _prompt(RAG_DOCS)

    if selected_option.action == Action.ACCEPT:
        console.print("[green]✓ Accepting all documents[/green]")
        return docs
    elif selected_option.action == Action.REJECT:
        console.print("[red]✗ Rejecting all documents[/red]")
        return []

    # Interactive review mode
    console.print("\n[bold]Document Review Mode[/bold]")
    filtered_docs = []

    for i, doc in enumerate(docs):
        section_path = get_section_path(doc)
        file_source = get_file_source(doc)
        content = format_doc(doc)
        content_within_julia = (
            content if not edit_julia_file else f"```julia\n{content.strip()}\n```"
        )

        # Create a table to show document info
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan")
        table.add_column("Value")
        table.add_row("Document", f"{i + 1}/{len(docs)}")
        table.add_row("Source", file_source)
        table.add_row("Section", section_path)

        console.print(f"\n{table}")
        utils.print_to_console(
            text=content_within_julia[:500] + "..."
            if len(content_within_julia) > 500
            else content_within_julia,
            title="Content",
            border_style=colorscheme.human_interaction,
        )

        console.print(
            "\nOptions: [bold](k)[/bold]eep | [bold](e)[/bold]dit | [bold](s)[/bold]kip | [bold](v)[/bold]iew-full"
        )
        doc_choice = Prompt.ask(
            "What to do with this document?", choices=["k", "e", "s", "v"], default="k"
        )

        if doc_choice == "k":
            filtered_docs.append(doc)
            console.print("[green]✓ Document kept[/green]")

        elif doc_choice == "s":
            console.print("[red]✗ Document skipped[/red]")

        elif doc_choice == "v":
            utils.print_to_console(
                text=content_within_julia,
                title="Full Document Content",
                border_style=colorscheme.success,
            )

            # Ask again after viewing
            console.print(
                "\nOptions: [bold](k)[/bold]eep | [bold](e)[/bold]dit | [bold](s)[/bold]kip"
            )
            doc_choice = Prompt.ask(
                "Now what to do with this document?",
                choices=["k", "e", "s"],
                default="k",
            )
            if doc_choice == "k":
                filtered_docs.append(doc)
                console.print("[green]✓ Document kept[/green]")
            elif doc_choice == "e":
                new_content = utils.edit_document_content(
                    content, edit_julia_file=edit_julia_file
                )
                if new_content.strip():
                    if edit_julia_file:
                        new_content = f"```julia\n{new_content.strip()}\n```"
                    filtered_docs.append(modify_doc_content(doc, new_content))
                    console.print("[green]✓ Document edited and kept[/green]")
                else:
                    console.print("[red]✗ Document removed (empty content)[/red]")
            else:  # doc_choice == "s"
                console.print("[red]✗ Document skipped[/red]")

        elif doc_choice == "e":
            new_content = utils.edit_document_content(
                content, edit_julia_file=edit_julia_file
            )
            if new_content.strip():
                if edit_julia_file:
                    new_content = f"```julia\n{new_content.strip()}\n```"
                filtered_docs.append(modify_doc_content(doc, new_content))
                console.print("[green]✓ Document edited and kept[/green]")
            else:
                console.print("[red]✗ Document removed (empty content)[/red]")

    console.print(
        f"\n[bold]Summary:[/bold] Kept {len(filtered_docs)}/{len(docs)} documents"
    )
    return filtered_docs


def response_on_check_code(code: str) -> tuple[bool, str, str]:
    """
    Returns:
        bool: Whether the user wants to check the code or not
        str: Additional feedback to the model
        str: The code to check (potentially edited)
    """
    selected_option = _prompt(CHECK_CODE)

    if selected_option.action == Action.ACCEPT:
        console.print("[green]✓ Running code checks[/green]")
        return True, "", code
    elif selected_option.action == Action.FEEDBACK:
        console.print("[bold blue]Give feedback:[/bold blue] ")
        user_input = console.input("> ")
        if not user_input.strip():  # If the user input is empty
            console.print("[red]✗ User feedback empty[/red]")
            return False, "", code
        console.print("[green]✓ Feedback received[/green]")

        # Log the feedback text
        logger = get_session_logger()
        if logger:
            logger.log_interaction(
                CHECK_CODE,
                selected_option.action,
                selected_option.label,
                user_input=user_input,
            )

        return False, user_input, code
    elif selected_option.action == Action.EDIT:
        console.print("\n[bold]Edit Code[/bold]")
        new_code = utils.edit_document_content(code, edit_julia_file=True)

        if new_code.strip():
            utils.print_to_console(
                text=add_julia_context(new_code),
                title="Code update",
                border_style=colorscheme.message,
            )
            console.print("[green]✓ Code updated[/green]")
            return True, "", new_code
        console.print("[red]✗ Code empty. Not updating![/red]")
        return True, "", code
    else:  # SKIP
        console.print("[red]✗ Skipping code checks[/red]")
        return False, "", code


def response_on_error() -> tuple[bool, str]:
    """
    Returns:
        bool: Whether the user wants to try fixing the code
        str: Additional feedback to the model
    """
    selected_option = _prompt(ON_ERROR)

    if selected_option.action == Action.ACCEPT:
        console.print("[green]✓ Trying to fix code[/green]")
        return True, ""
    elif selected_option.action == Action.FEEDBACK:
        console.print("[bold blue]Give feedback:[/bold blue]")
        user_input = console.input("> ")
        if not user_input.strip():  # If the user input is empty
            console.print("[red]✗ User feedback empty[/red]")
            return True, ""
        console.print("[green]✓ Feedback received[/green]")

        # Log the feedback text
        logger = get_session_logger()
        if logger:
            logger.log_interaction(
                ON_ERROR,
                selected_option.action,
                selected_option.label,
                user_input=user_input,
            )

        return True, user_input
    else:  # SKIP
        console.print("[red]✗ Skipping code fix[/red]")
        return False, ""


def modify_rag_query(query: str, retriever_name: str) -> str:
    """
    CLI version of modify_rag_query that allows interactive query modification.

    Args:
        query: The original query string
        retriever_name: Name of the retriever (e.g., "JutulDarcy")

    Returns:
        str: The potentially modified query (empty string if skipped)
    """
    utils.print_to_console(
        text=f"**Original Query:** `{query}`",
        title=f"Retrieving from {retriever_name}",
        border_style=colorscheme.warning,
    )

    selected_option = _prompt(RAG_QUERY)

    if selected_option.action == Action.ACCEPT:
        console.print(f"[green]✓ Using original query for {retriever_name}[/green]")
        return query
    elif selected_option.action == Action.EDIT:
        new_query = utils.edit_document_content(query)

        if new_query.strip():
            console.print(f"[green]✓ Query updated for {retriever_name}[/green]")
            utils.print_to_console(
                text=f"**New Query:** `{new_query.strip()}`",
                title="Updated Query",
                border_style=colorscheme.success,
            )
            return new_query.strip()
        else:
            console.print("[yellow]⚠ Empty query, using original[/yellow]")
            return query
    else:  # SKIP
        console.print(f"[red]✗ Skipping {retriever_name} retrieval[/red]")
        return ""  # Return empty string to indicate no query


def modify_terminal_run(command: str) -> tuple[bool, str]:
    """
    Accept, modify, or skip a terminal command.

    Args:
        command: The original terminal command string

    Returns:
        tuple[bool, str]: (should_run, command)
            - should_run: Whether the command is allowed to run
            - command: The potentially modified command
    """
    utils.print_to_console(
        text=f"**Command:** `{command}`",
        title="Trying to run in terminal",
        border_style=colorscheme.warning,
    )

    selected_option = _prompt(TERMINAL_RUN)

    if selected_option.action == Action.ACCEPT:
        console.print("[green]✓ Running original command[/green]")
        return True, command
    elif selected_option.action == Action.EDIT:
        new_command = utils.edit_document_content(command)

        if new_command.strip():
            console.print("[green]✓ Running updated command[/green]")
            return True, new_command
        else:
            console.print("[yellow]⚠ Empty command. Not running[/yellow]")
            return False, command
    else:  # SKIP
        console.print("[red]✗ Skipping running command[/red]")
        return False, command
