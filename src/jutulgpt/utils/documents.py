"""Document processing utilities."""

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
