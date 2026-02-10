"""LangGraph UI implementations for human-in-the-loop interactions.

This module consolidates all UI handlers that use LangGraph's HumanInterrupt
for user interactions. All handlers use shared interaction definitions from
interactions.py for consistency and logging.
"""

from typing import Callable, List, Optional

from langchain_core.documents import Document
from langgraph.prebuilt.interrupt import (
    ActionRequest,
    HumanInterrupt,
    HumanInterruptConfig,
    HumanResponse,
)
from langgraph.types import interrupt

from jutulgpt.logging.session import get_session_logger
from jutulgpt.rag.utils import modify_doc_content
from jutulgpt.utils.code_parsing import add_julia_context, remove_julia_context

from .interactions import (
    Action,
    FILE_WRITE,
    ON_ERROR,
    RAG_DOCS,
    RAG_QUERY,
    CHECK_CODE,
    Interaction,
)


# Map HumanInterrupt response types to our unified Action enum
_RESPONSE_TO_ACTION = {
    "accept": Action.ACCEPT,
    "edit": Action.EDIT,
    "ignore": Action.SKIP,
    "respond": Action.FEEDBACK,
}


def _response_to_action(response_type: str) -> Action:
    """Convert HumanInterrupt response type to Action enum."""
    return _RESPONSE_TO_ACTION.get(response_type, Action.SKIP)


def _build_option_labels(interaction: Interaction) -> list[str]:
    """Build UI-friendly option labels from interaction.

    Maps our semantic actions to HumanInterrupt's button descriptions.

    Returns:
        List of formatted option labels (e.g., "Accept to check the code")
    """
    labels = []
    for opt in interaction.options:
        if opt.action == Action.ACCEPT:
            labels.append(f"Accept to {opt.label.lower()}")
        elif opt.action == Action.EDIT:
            labels.append(f"Edit to {opt.label.lower()}")
        elif opt.action == Action.SKIP or opt.action == Action.REJECT:
            labels.append(f"Ignore to {opt.label.lower()}")
        elif opt.action == Action.FEEDBACK:
            labels.append(f"Respond to {opt.label.lower()}")
    return labels


def _build_action_text(interaction: Interaction) -> str:
    """Build HumanInterrupt action description from interaction options.

    Maps our semantic actions to HumanInterrupt's button descriptions.
    """
    parts = _build_option_labels(interaction)
    return ". ".join(parts) + "."


def response_on_check_code(code: str) -> tuple[bool, str, str]:
    """Prompt user to check, edit, skip, or provide feedback on generated code.

    Returns:
        tuple[bool, str, str]: (should_check, feedback, code)
            - should_check: Whether to proceed with code checking
            - feedback: User feedback string (empty if not provided)
            - code: The code to check (potentially edited)
    """
    request = HumanInterrupt(
        action_request=ActionRequest(
            action=_build_action_text(CHECK_CODE),
            args={"Code": add_julia_context(code)},
        ),
        config=HumanInterruptConfig(
            allow_ignore=True,
            allow_accept=True,
            allow_respond=True,
            allow_edit=True,
        ),
    )

    human_response: HumanResponse = interrupt([request])[0]
    response_type = human_response.get("type")
    action = _response_to_action(response_type)

    # Log the interaction
    logger = get_session_logger()
    if logger:
        logger.log_interaction(CHECK_CODE, action)

    if action == Action.ACCEPT:
        return True, "", code
    elif action == Action.EDIT:
        args_dict = human_response.get("args", {}).get("args", {})
        new_code = args_dict.get("Code", code)
        new_code = remove_julia_context(new_code)
        if new_code.strip():
            code = new_code
        return True, "", code
    elif action == Action.FEEDBACK:
        # Note: "respond" type is not fully implemented in LangGraph UI yet
        # For now, treat as skip
        raise NotImplementedError(
            f"Interrupt value of type {response_type} is not yet implemented."
        )
    elif response_type == Action.SKIP:
        return False, "", code
    else:
        raise TypeError(f"Interrupt value of type {response_type} is not supported.")


def response_on_error() -> tuple[bool, str]:
    """Prompt user for action when code check fails.

    Returns:
        tuple[bool, str]: (should_fix, feedback)
            - should_fix: Whether to attempt fixing the code
            - feedback: Additional feedback for the model (empty if not provided)
    """
    request = HumanInterrupt(
        action_request=ActionRequest(
            action=_build_action_text(ON_ERROR),
            args={},
        ),
        config=HumanInterruptConfig(
            allow_ignore=True,
            allow_accept=True,
            allow_respond=True,
            allow_edit=False,
        ),
    )

    # Wait for the user's response from the UI
    human_response: HumanResponse = interrupt([request])[0]
    response_type = human_response.get("type")
    action = _response_to_action(response_type)

    # Log the interaction
    logger = get_session_logger()
    if logger:
        logger.log_interaction(ON_ERROR, action)

    if action == Action.ACCEPT:
        return True, ""
    elif action == Action.FEEDBACK:
        # Note: "respond" type is not fully implemented in LangGraph UI yet
        raise NotImplementedError(
            f"Interrupt value of type {response_type} is not yet implemented."
        )
    elif response_type == Action.SKIP:
        return False, ""
    else:
        raise TypeError(f"Interrupt value of type {response_type} is not supported.")


def modify_rag_query(query: str, retriever_name: str) -> str:
    """Prompt user to modify or skip a RAG query.

    Args:
        query: The original query string
        retriever_name: Name of the retriever (e.g., "JutulDarcy")

    Returns:
        str: The potentially modified query (empty string if skipped)
    """
    # Custom action text for RAG query (includes retriever name)
    action_text = f"Trying to retrieve documents from {retriever_name}. Modify the query if needed."

    request = HumanInterrupt(
        action_request=ActionRequest(
            action=action_text,
            args={"Query": query},
        ),
        config=HumanInterruptConfig(
            allow_ignore=False,  # Can't skip, but can edit or accept
            allow_accept=True,
            allow_respond=False,
            allow_edit=True,
        ),
    )

    # Wait for the user's response
    human_response: HumanResponse = interrupt([request])[0]
    response_type = human_response.get("type")
    action = _response_to_action(response_type)

    # Log the interaction
    logger = get_session_logger()
    if logger:
        logger.log_interaction(RAG_QUERY, action)

    if action == Action.EDIT:
        args_dict = human_response.get("args", {}).get("args", {})
        new_query = args_dict.get("Query", query)
        if new_query.strip():
            query = new_query
    elif action == Action.SKIP:
        # Map "ignore" to skip (return empty query)
        query = ""
    elif action == Action.ACCEPT:
        pass  # Use original query

    return query


def response_on_rag(
    docs: List[Document],
    get_file_source: Callable,
    get_section_path: Callable,
    format_doc: Callable,
    action_name: str = "Modify",
) -> List[Document]:
    """
    Presents retrieved RAG documents to the user for optional modification before sending them as LLM context.

    This function enables a human-in-the-loop workflow, allowing the user to review and edit the content of each document.
    If the user deletes all content for a document, that document is removed from the list. The function supports custom
    formatting and section/file labeling for each document.

    Args:
        docs (List[Document]): List of documents retrieved by the RAG system.
        get_file_source (Callable): Function to get the file source of a document (for UI display).
        get_section_path (Callable): Function to get the section path of a document (for UI display).
        format_doc (Callable): Function to format a document for display in the UI.
        action_name (str, optional): Name of the action to be displayed in the UI. Defaults to "Modify".

    Returns:
        List[Document]: The list of documents after potential modifications.
            Documents with empty content are removed.
    """
    if not docs:
        return docs  # Nothing to do if no documents

    action_request_args = {}
    arg_names = []

    # Build the arguments for the UI, ensuring unique names for each document
    for _, doc in enumerate(docs):
        section_path = get_section_path(doc)
        file_source = get_file_source(doc)
        arg_name = f"{file_source} - {section_path}"
        # Ensure arg_name is unique by appending a suffix if needed
        original_arg_name = arg_name
        suffix = 1
        while arg_name in action_request_args:
            arg_name = f"{original_arg_name} ({suffix})"
            suffix += 1
        arg_names.append(arg_name)
        content = format_doc(doc)
        action_request_args[arg_name] = content

    request = HumanInterrupt(
        action_request=ActionRequest(
            action=action_name,
            args=action_request_args,
        ),
        config=HumanInterruptConfig(
            allow_ignore=True,
            allow_accept=True,
            allow_respond=False,
            allow_edit=True,
        ),
    )

    human_response: HumanResponse = interrupt([request])[0]
    response_type = human_response.get("type")
    action = _response_to_action(response_type)

    # Log the interaction
    logger = get_session_logger()
    if logger:
        logger.log_interaction(RAG_DOCS, action)

    if action == Action.EDIT:
        # User edited one or more documents
        args_dict = human_response.get("args", {}).get("args", {})
        new_docs = []
        for i, (arg_name, doc) in enumerate(zip(arg_names, docs)):
            new_content = args_dict.get(arg_name, None)
            if new_content and new_content.strip():
                # Keep and update the document if content is not empty/whitespace
                new_docs.append(modify_doc_content(doc, new_content))
            # else: skip (remove) the doc
        docs = new_docs
    elif action == Action.ACCEPT:
        # If accepted, no changes to documents
        pass
    elif action == Action.SKIP or action == Action.REJECT:
        # If ignored/rejected, all retrieved documents are removed
        docs = []
    else:
        raise TypeError(f"Interrupt value of type {response_type} is not supported.")

    return docs


def response_on_file_write(file_path: str) -> tuple[bool, str]:
    """Prompt user when agent tries to write to an existing file.

    Returns:
        tuple[bool, str]: (should_write, file_path)
            - should_write: Whether to proceed with writing
            - file_path: The file path (potentially modified)
    """
    request = HumanInterrupt(
        action_request=ActionRequest(
            action=_build_action_text(FILE_WRITE),
            args={"Filepath": file_path},
        ),
        config=HumanInterruptConfig(
            allow_ignore=True,
            allow_accept=True,
            allow_respond=False,
            allow_edit=True,
        ),
    )

    # Wait for the user's response from the UI
    human_response: HumanResponse = interrupt([request])[0]
    response_type = human_response.get("type")
    action = _response_to_action(response_type)

    # Log the interaction
    logger = get_session_logger()
    if logger:
        logger.log_interaction(FILE_WRITE, action)

    if action == Action.ACCEPT:
        return True, file_path
    elif action == Action.EDIT:
        args_dict = human_response.get("args", {}).get("args", {})
        file_path = args_dict.get("Filepath", file_path)
        return True, file_path
    elif response_type == Action.SKIP:
        return False, file_path
    else:
        raise TypeError(f"Interrupt value of type {response_type} is not supported.")
