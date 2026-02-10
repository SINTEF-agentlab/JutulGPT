"""Utility & helper functions.

Submodules:
- code_parsing: Extracting and parsing code from LLM responses
- code_transforms: Transforming Julia code before execution
- documents: Document processing utilities
- model: Model loading and message utilities
"""

from jutulgpt.utils.code_parsing import (
    _get_code_string_from_response,
    add_julia_context,
    format_code_response,
    get_code_from_response,
    get_last_code_response,
    remove_julia_context,
    split_code_into_lines,
)
from jutulgpt.utils.code_transforms import (
    check_for_package_install,
    remove_plotting,
    replace_case,
    shorten_first_argument,
    shorter_simulations,
)
from jutulgpt.utils.documents import (
    _get_relevant_part_of_file_source,
    deduplicate_document_chunks,
    get_file_source,
)
from jutulgpt.utils.model import (
    get_message_text,
    get_provider_and_model,
    get_tool_message,
    load_chat_model,
    trim_state_messages,
)

__all__ = [
    # code_parsing
    "_get_code_string_from_response",
    "add_julia_context",
    "format_code_response",
    "get_code_from_response",
    "get_last_code_response",
    "remove_julia_context",
    "split_code_into_lines",
    # code_transforms
    "check_for_package_install",
    "remove_plotting",
    "replace_case",
    "shorten_first_argument",
    "shorter_simulations",
    # documents
    "_get_relevant_part_of_file_source",
    "deduplicate_document_chunks",
    "get_file_source",
    # model
    "get_message_text",
    "get_provider_and_model",
    "get_tool_message",
    "load_chat_model",
    "trim_state_messages",
]
