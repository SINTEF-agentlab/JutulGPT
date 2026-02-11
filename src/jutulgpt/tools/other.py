from __future__ import annotations

import os
import re

from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field
from rich.panel import Panel
from typing_extensions import Annotated

from jutulgpt.cli import colorscheme, print_to_console
from jutulgpt.configuration import DISPLAY_CONTENT_MAX_LENGTH, cli_mode
from jutulgpt.globals import console
from jutulgpt.logging import ToolEntry, get_session_logger


def _truncate(text: str, max_length: int = DISPLAY_CONTENT_MAX_LENGTH) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _get_fence(content: str, min_backticks: int = 3) -> str:
    """Return a fence string that won't conflict with content.

    Finds the longest sequence of backticks in content and returns
    a fence that is at least one backtick longer.
    """
    matches = re.findall(r"`+", content)
    max_found = max((len(m) for m in matches), default=0)
    num_backticks = max(min_backticks, max_found + 1)
    return "`" * num_backticks


class ReadFromFileInput(BaseModel):
    file_path: str = Field(description="The absolute path of the file to read.")
    read_full_file: bool = Field(
        description="Whether to read the full file (ignoring line range)."
    )
    start_line_number_base_zero: int = Field(
        description="The line number to start reading from, 0-based."
    )
    end_line_number_base_zero: int = Field(
        description="The inclusive line number to end reading at, 0-based."
    )


@tool(
    "read_from_file",
    description="Read file contents. Has the option to specify the line range. Returns a string containing the specified lines or the entire file.",
    args_schema=ReadFromFileInput,
)
def read_from_file(
    file_path: str,
    read_full_file: bool,
    start_line_number_base_zero: int,
    end_line_number_base_zero: int,
) -> str:
    try:
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Convert to 0-based indexing
        if read_full_file:
            start = 0
            end = len(lines)
        else:
            start = max(0, start_line_number_base_zero)
            end = min(len(lines), end_line_number_base_zero + 1)

        if start >= len(lines):
            return f"Start line {start_line_number_base_zero} is beyond file length ({len(lines)} lines)"

        selected_lines = lines[start:end]

        # Add line numbers
        result_lines = []
        for i, line in enumerate(selected_lines, start=start):
            result_lines.append(f"{i:4d}: {line.rstrip()}")

        total_lines = len(lines)

        content = "\n".join(result_lines)

        summary = f"Reading file {file_path} (lines {start}-{end - 1})"
        print_to_console(
            text=summary,
            title="Read file",
            border_style=colorscheme.message,
        )
        logger = get_session_logger()
        if logger:
            logger.log(
                ToolEntry(
                    content=summary,
                    title="Read file",
                    tool_name="read_from_file",
                    args={"file_path": file_path, "lines": f"{start}-{end - 1}"},
                )
            )
        return (
            f"File: {file_path} (lines {start}-{end - 1} of {total_lines} total)\n"
            + "\n".join(result_lines)
        )

    except Exception as e:
        return f"Error reading file: {str(e)}"


class WriteToFileInput(BaseModel):
    file_path: str = Field(
        "The absolute path to the file to write to",
    )
    content: str = Field(
        "The content to write to the file",
    )


@tool(
    "write_to_file",
    description="Write a string to file.",
    args_schema=WriteToFileInput,
)
def write_to_file(
    file_path: str,
    content: str,
    config: Annotated[RunnableConfig, InjectedToolArg] = None,
) -> str:
    from jutulgpt.configuration import BaseConfiguration

    # Check if we should skip approval (for autonomous mode)
    config = ensure_config(config)
    configuration = BaseConfiguration.from_runnable_config(config)
    skip_approval = configuration.skip_terminal_approval

    # Check if file already exists
    if os.path.exists(file_path) and not skip_approval:
        try:
            # Read existing content for preview
            with open(file_path, "r", encoding="utf-8") as f:
                existing_content = f.read()

            if cli_mode:
                # Create preview of existing content (first 300 chars)
                existing_preview = existing_content[:300]
                if len(existing_content) > 300:
                    existing_preview += "..."

                # Create preview of new content (first 300 chars)
                new_preview = content[:300]
                if len(content) > 300:
                    new_preview += "..."

                # Display confirmation prompt with previews
                console.print(
                    Panel.fit(
                        f"[bold yellow]File Already Exists: {file_path}[/bold yellow]\n\n"
                        f"[bold]Current content ({len(existing_content)} chars):[/bold]\n"
                        f"[dim]{existing_preview}[/dim]\n\n"
                        f"[bold]New content ({len(content)} chars):[/bold]\n"
                        f"[dim]{new_preview}[/dim]",
                        title="File Overwrite Confirmation",
                        border_style="yellow",
                    )
                )

                # Prompt for confirmation
                response = (
                    console.input(
                        "\n[bold red]Do you want to overwrite this file? (y/n): [/bold red]"
                    )
                    .strip()
                    .lower()
                )

                if response not in ["y", "yes"]:
                    print_to_console(
                        text=f"File write cancelled by user: {file_path}",
                        title="File Writer",
                        border_style=colorscheme.warning,
                    )
                    return f"File write cancelled by user: {file_path}"
            else:  # UI mode
                from jutulgpt.human_in_the_loop.ui import response_on_file_write

                write_to_file, file_path = response_on_file_write(file_path)
                if not write_to_file:
                    return f"File write cancelled by user: {file_path}"

        except Exception as e:
            print_to_console(
                text=f"Error reading existing file {file_path}: {str(e)}",
                title="File Writer Warning",
                border_style=colorscheme.warning,
            )

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            success_msg = f"Successfully wrote to file: {file_path}"
            print_to_console(
                text=success_msg,
                title="File Writer Success",
                border_style=colorscheme.success,
            )
            return success_msg
    except Exception as e:
        error_msg = f"Error writing to file {file_path}: {str(e)}"
        print_to_console(
            text=error_msg,
            title="File Writer Error",
            border_style=colorscheme.error,
        )
        return error_msg


@tool("get_working_directory", description=" Get the current working directory path.")
def get_working_directory() -> str:
    return os.getcwd()


class ListFilesInDocumentationInput(BaseModel):
    directory_path: str = Field(
        description="The absolute path of the directory to list files from."
    )
    recursive: bool = Field(
        description="True to list files recursively, False to list only top-level files."
    )


@tool(
    "list_files_in_directory",
    description="Recursievly list all files in a directory. Returns a string with the absolute paths of all files and directories.",
    args_schema=ListFilesInDocumentationInput,
)
def list_files_in_directory(directory_path: str, recursive: bool) -> str:
    try:
        if not os.path.exists(directory_path):
            return f"ERROR: Directory {directory_path} does not exist."

        if not os.path.isdir(directory_path):
            return f"ERROR: {directory_path} is not a directory."

        file_paths = []
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    path = os.path.join(root, file)
                    file_paths.append(f"[FILE] {path}")
                for dir in dirs:
                    path = os.path.join(root, dir)
                    file_paths.append(f"[DIR]  {path}/")
        else:
            # Only list top-level files and directories
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                if os.path.isdir(item_path):
                    file_paths.append(f"[DIR]  {item_path}/")
                else:
                    file_paths.append(f"[FILE] {item_path}")

        if not file_paths:
            return f"No files found in directory: {directory_path}"

        file_paths.sort()
        mode = "recursive" if recursive else "top-level"
        return f"Contents of {directory_path} ({mode}):\n" + "\n".join(file_paths)

    except Exception as e:
        return f"ERROR: Failed to list directory contents: {str(e)}"
