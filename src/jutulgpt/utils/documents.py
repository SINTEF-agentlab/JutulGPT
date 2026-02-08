"""Document processing utilities."""

import os
from typing import List

from langchain_core.documents import Document


def deduplicate_document_chunks(chunks: List[Document]) -> List[Document]:
    """
    Remove duplicate Document chunks based on their page content.

    Args:
        chunks (List[Document]): List of Document objects.

    Returns:
        List[Document]: List of unique Document objects.
    """
    seen = set()
    deduped = []
    for doc in chunks:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            deduped.append(doc)
    return deduped


def get_file_source(doc: Document) -> str:
    file_source = doc.metadata.get("source", "Unknown Document")
    return file_source


def _get_relevant_part_of_file_source(source: str, relevant_doc_name: str = "rag"):
    """
    Remove the part of the soure up to and including the relevant_doc_name.
    """
    idx = source.find(f"/{relevant_doc_name}/")
    if idx != -1:
        source = source[idx + len("/rag/") :]
    return source


def load_lines_from_txt(file_path: str) -> List[str]:
    """
    Load lines from a text file, stripping whitespace and ignoring empty lines.

    Args:
        file_path (str): Path to the text file.

    Returns:
        list: List of non-empty, stripped lines from the file.
    """
    if not file_path:
        raise ValueError("File path cannot be empty.")
    if not isinstance(file_path, str):
        file_path = str(file_path)
    try:
        with open(file_path, "r") as file:
            return [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The file at {file_path} does not exist. Current working directory is {os.getcwd()}."
        )
    except IOError as e:
        raise IOError(
            f"An error occurred while reading the file at {file_path}: {e}"
        ) from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}") from e
