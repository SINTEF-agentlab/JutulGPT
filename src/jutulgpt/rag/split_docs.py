import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from jutulgpt.utils.documents import deduplicate_document_chunks, get_file_source


def split_docs(
    document: Document,
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    content = document.page_content
    document_metadata = document.metadata.copy()

    # Some processing
    content = preprocess_content(content)

    # Split documents
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True,
    )
    splits = markdown_splitter.split_text(content)

    processed_docs = []
    for split in splits:
        split.metadata = {**split.metadata, **document_metadata}

        # Clean up header values by removing '{#...}' if present
        for key in ["Header 1", "Header 2", "Header 3"]:
            if key in split.metadata and split.metadata[key]:
                # Remove '{#...}' from the header value
                split.metadata[key] = re.sub(
                    r"\s*\{#[^}]*\}", "", split.metadata[key]
                ).strip()

        processed_docs.append(split)

    # Merge small splits for minimum context size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    final_docs = []
    for doc in text_splitter.split_documents(processed_docs):
        doc.metadata.update(document_metadata)  # reapply original metadata
        final_docs.append(doc)

    return final_docs


def preprocess_content(content: str) -> str:
    # Remove blockquotes
    content = re.sub(r"^\s*>+", "", content, flags=re.MULTILINE).strip()
    # Remove images
    content = re.sub(r"!\[.*?\]\(.*?\)", "", content).strip()
    # Remove ```ansi blocks
    content = re.sub(r"```ansi[\s\S]*?```", "", content, flags=re.MULTILINE).strip()
    return content


def remove_markdown_links(text: str) -> str:
    # Replace [text](url) with just text
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)


def format_doc(doc: Document) -> str:
    page_content = doc.page_content.strip()
    return f"{page_content}"


def get_section_path(doc: Document, for_ui_printing: bool = False) -> str:
    header_keys = ["Header 1", "Header 2", "Header 3"]
    section_path_parts = [
        str(doc.metadata[k])
        for k in header_keys
        if k in doc.metadata and doc.metadata[k] is not None
    ]

    if for_ui_printing:
        section_path = section_path_parts[0] if section_path_parts else "Root"
        return section_path
    section_path = " > ".join(section_path_parts) if section_path_parts else "Root"
    return section_path


def format_docs(docs, remove_duplicates: bool = True):
    if remove_duplicates:
        docs = deduplicate_document_chunks(docs)

    formatted = []
    for doc in docs:
        doc_string = ""
        file_source = get_file_source(doc)
        section_path = get_section_path(doc)
        doc_string += f"# From `{file_source}`: Section `{section_path}`\n"
        doc_string += f"{format_doc(doc)}"
        formatted.append(doc_string)
    return "\n\n".join(formatted)
