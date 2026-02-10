"""CLI utilities for console output and user interaction.

This module provides Rich console rendering utilities for the JutulGPT CLI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from jutulgpt.configuration import SHOW_REASONING_SUMMARY
from jutulgpt.globals import console
from jutulgpt.state import CodeBlock

if TYPE_CHECKING:
    from jutulgpt.logging import SessionLogger


def print_to_console(
    text: str,
    title: str = "Assistant",
    border_style: str = "",
    panel_kwargs: Optional[dict] = None,
    with_markdown: bool = True,
):
    """Print text to the console with a Rich panel.

    Args:
        text: The text to print.
        title: The title of the panel.
        border_style: Style for panel border.
        panel_kwargs: Additional keyword arguments for the panel.
        with_markdown: Whether to render text as markdown.
    """
    panel_kwargs = panel_kwargs.copy() if panel_kwargs else {}  # prevent mutation

    if border_style:
        panel_kwargs["border_style"] = border_style
    if title:
        panel_kwargs["title"] = title

    console.print(Panel.fit(Markdown(text) if with_markdown else text, **panel_kwargs))


def _extract_text_from_chunk(chunk: Any) -> str:
    """Extract human-readable text from a LangChain streaming chunk."""
    txt = getattr(chunk, "text", None)
    if isinstance(txt, str) and txt:
        return txt

    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                t = block.get("text")
                if isinstance(t, str) and t:
                    texts.append(t)
        return "".join(texts)

    return ""


def _extract_reasoning_summary(msg: Any) -> str:
    """Extract reasoning summary from OpenAI Responses API.

    The summary is in msg.content as:
    {'type': 'reasoning', 'summary': [{'type': 'summary_text', 'text': '...'}]}
    """
    parts: list[str] = []

    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return ""

    for block in content:
        if not isinstance(block, dict):
            continue

        # Look for reasoning blocks with summary
        if block.get("type") == "reasoning" or "summary" in block:
            summary = block.get("summary")
            if isinstance(summary, list):
                for item in summary:
                    if isinstance(item, dict):
                        txt = item.get("text")
                        if isinstance(txt, str) and txt.strip():
                            parts.append(txt.strip())

    return "\n\n".join(parts).strip()


def stream_to_console(
    llm,
    message_list: List,
    config: RunnableConfig,
    title: Optional[str] = "",
    border_style: str = "",
    panel_kwargs: Optional[dict] = None,
    with_markdown: bool = True,
    logger: Optional["SessionLogger"] = None,
    log_kwargs: Optional[dict] = None,
) -> AIMessage:
    """Stream LLM response to console with live tail view.

    Uses height-constrained panels during streaming to prevent overflow.
    Full formatted output is printed to scrollback when streaming completes.

    Phases:
    - reasoning: Shows tail of reasoning summary (yellow panel)
    - text: Shows tail of agent output with character count
    - transition: Prints reasoning to scrollback when text begins
    - end: Prints full formatted markdown output to scrollback

    Note: Resizing the terminal during streaming may cause display artifacts.
    This is a known limitation of Rich's Live display.
    """
    ai_message: Optional[AIMessage] = None
    reasoning_summary = ""
    streamed_text = ""

    panel_kwargs = panel_kwargs.copy() if panel_kwargs else {}
    if border_style:
        panel_kwargs["border_style"] = border_style
    if title:
        panel_kwargs["title"] = title

    live: Optional[Live] = None
    phase = "idle"  # idle -> reasoning -> text

    def _get_max_height() -> int:
        """Get max panel height based on current console size."""
        return max(8, (console.size.height - 4) // 2)

    def _tail_lines(text: str, max_lines: int) -> str:
        """Get last N lines of text."""
        lines = text.splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[-max_lines:])
        return text

    def _build_display() -> Panel:
        """Build streaming display panel (grows with content, limited by tail view)."""
        max_h = _get_max_height()
        if phase == "reasoning" and reasoning_summary:
            return Panel(
                Text(_tail_lines(reasoning_summary, max_h), overflow="ellipsis"),
                title="Reasoning Summary",
                border_style="yellow",
            )
        elif phase == "text" and streamed_text:
            char_count = len(streamed_text)
            display_title = f"{title or 'Agent'} [dim]({char_count:,} chars)[/dim]"
            return Panel(
                Text(_tail_lines(streamed_text, max_h), overflow="ellipsis"),
                title=Text.from_markup(display_title),
                border_style=border_style or "cyan",
            )
        return Panel(Text("Waiting..."), title="Agent", border_style="dim")

    for chunk in llm.stream(message_list, config=config):
        ai_message = chunk if ai_message is None else ai_message + chunk

        # Extract reasoning summary
        if SHOW_REASONING_SUMMARY and ai_message:
            new_summary = _extract_reasoning_summary(ai_message)
            if new_summary:
                reasoning_summary = new_summary

        # Extract text
        text_part = _extract_text_from_chunk(chunk)
        if text_part:
            streamed_text += text_part

        # Determine phase
        if streamed_text:
            new_phase = "text"
        elif reasoning_summary:
            new_phase = "reasoning"
        else:
            new_phase = "idle"

        # Handle phase transition: reasoning -> text
        if phase == "reasoning" and new_phase == "text":
            # Stop live, print reasoning to scrollback
            if live is not None:
                live.stop()
                live = None
            if reasoning_summary:
                console.print(
                    Panel.fit(
                        Markdown(reasoning_summary),
                        title="Reasoning Summary",
                        border_style="yellow",
                    )
                )

        phase = new_phase

        # Update live display
        if phase != "idle":
            if live is None:
                live = Live(
                    _build_display(),
                    console=console,
                    transient=True,
                    refresh_per_second=8,
                )
                live.start()
            else:
                live.update(_build_display())

    # Cleanup live
    if live is not None:
        live.stop()

    # Get final summary
    if SHOW_REASONING_SUMMARY and ai_message:
        final = _extract_reasoning_summary(ai_message)
        if final:
            reasoning_summary = final

    # Log to file
    if logger and logger.enabled:
        kwargs = log_kwargs.copy() if log_kwargs else {}
        # Add tool calls from response if present
        if ai_message and getattr(ai_message, "tool_calls", None):
            kwargs["tool_calls"] = ai_message.tool_calls
        logger.log_assistant(
            content=streamed_text,
            title=title or "Assistant",
            reasoning_summary=reasoning_summary if SHOW_REASONING_SUMMARY else None,
            **kwargs,
        )

    # Print reasoning summary to scrollback if present
    has_text = bool(streamed_text.strip())
    has_tools = ai_message and getattr(ai_message, "tool_calls", None)

    # Reasoning: only if we stayed in reasoning phase (tool-call-only response)
    # If we transitioned to text, reasoning was already printed during the transition
    if SHOW_REASONING_SUMMARY and reasoning_summary.strip() and phase != "text":
        console.print(
            Panel.fit(
                Markdown(reasoning_summary),
                title="Reasoning Summary",
                border_style="yellow",
            )
        )

    # Agent text output (streaming only showed tail view)
    if has_text:
        content = Markdown(streamed_text) if with_markdown else streamed_text
        console.print(Panel.fit(content, **panel_kwargs))

    if ai_message is None:
        raise RuntimeError("No message from model")
    return ai_message


def show_startup_screen():
    """Display the JutulGPT startup screen with ASCII art."""
    subtitle = Text(
        "SINTEF Digital's AI Assistant for JutulDarcy",
        justify="center",
        style="italic green",
    )

    info_text = Text.from_markup(
        "\n[bold cyan]Type your prompt below, or type [yellow]'q'[/yellow] to quit.[/bold cyan]"
    )

    ascii_art = Text.from_markup(
        "[bold green]"
        "     ██╗██╗   ██╗████████╗██╗   ██╗██╗      ██████╗ ██████╗ ████████╗\n"
        "     ██║██║   ██║╚══██╔══╝██║   ██║██║     ██╔════╝ ██╔══██╗╚══██╔══╝\n"
        "     ██║██║   ██║   ██║   ██║   ██║██║     ██║  ███╗██████╔╝   ██║   \n"
        "██   ██║██║   ██║   ██║   ██║   ██║██║     ██║   ██║██╔═══╝    ██║   \n"
        "╚█████╔╝╚██████╔╝   ██║   ╚██████╔╝███████╗╚██████╔╝██║        ██║   \n"
        " ╚════╝  ╚═════╝    ╚═╝    ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝        ╚═╝   \n"
        "[/bold green]"
    )

    content = Group(
        Align.center(ascii_art),
        Align.center(subtitle),
        Align.center(info_text),
    )

    panel = Panel.fit(
        content,
        border_style="green",
        padding=(1, 4),
        title="",
        title_align="left",
    )
    console.print(panel)


def edit_document_content(original_content: str, edit_julia_file: bool = False) -> str:
    """Allow user to edit document content in an external editor.

    Args:
        original_content: The original document content.
        edit_julia_file: If True, use .jl extension; otherwise .md.

    Returns:
        The edited content.
    """
    import os
    import subprocess
    import tempfile

    file_suffix = ".jl" if edit_julia_file else ".md"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=file_suffix, delete=False
        ) as f:
            f.write(original_content)
            f.flush()

            editor = os.environ.get("EDITOR", "vim")
            try:
                subprocess.run([editor, f.name], check=True)

                with open(f.name, "r") as edited_file:
                    edited_content = edited_file.read()

                os.unlink(f.name)
                return edited_content

            except subprocess.CalledProcessError:
                console.print(
                    f"[red]Error opening editor '{editor}'. Falling back to original content.[/red]"
                )
                os.unlink(f.name)
                return original_content
            except FileNotFoundError:
                console.print(
                    f"[red]Editor '{editor}' not found. Try setting EDITOR environment variable.[/red]"
                )
                os.unlink(f.name)
                return original_content

    except Exception as e:
        console.print(f"[red]Error with external editor: {e}[/red]")
        return original_content


def save_code_to_file(code_block: CodeBlock) -> None:
    """
    Helper function to save code block to file with user interaction.

    Args:
        code_block: The code block to save
    """
    import os

    console.print("\n[bold yellow]Save Code to File[/bold yellow]")

    # Ask for filename
    default_filename = "generated_code.jl"
    filename = Prompt.ask("Enter filename", default=default_filename)

    # Ensure .jl extension
    if not filename.endswith(".jl"):
        filename += ".jl"

    try:
        # Check if file exists and ask for confirmation
        if os.path.exists(filename):
            overwrite = Prompt.ask(
                f"File '{filename}' already exists. Overwrite?",
                choices=["y", "n"],
                default="n",
            )
            if overwrite.lower() != "y":
                console.print("[yellow]⚠ File save cancelled[/yellow]")
                return

        # Write the code to file
        with open(filename, "w") as f:
            if code_block.imports:
                f.write(code_block.imports + "\n\n")
            f.write(code_block.code)

        console.print(f"[green]✓ Code saved to '{filename}' successfully[/green]")

    except Exception as e:
        console.print(f"[red]✗ Error saving file: {str(e)}[/red]")
