from langchain_community.retrievers import BM25Retriever

from jutulgpt.rag.retriever_specs import RetrieverSpec


def _load_and_split_docs(spec: RetrieverSpec) -> list:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    # Resolve dir_path if it's a callable
    dir_path = spec.dir_path() if callable(spec.dir_path) else spec.dir_path

    filetypes = [spec.filetype] if isinstance(spec.filetype, str) else spec.filetype

    docs = []
    for filetype in filetypes:
        loader = DirectoryLoader(
            path=dir_path,
            glob=f"**/*.{filetype}",
            show_progress=False,
            loader_cls=TextLoader,
        )
        docs.extend(loader.load())

    # Split documents
    chunks = []
    for doc in docs:
        chunks.extend(spec.split_func(doc))
    return chunks


def make_retriever(spec: RetrieverSpec, k: int = 3) -> BM25Retriever:
    """Create a BM25 retriever from the given spec.

    Args:
        spec: The retriever specification defining docs to load.
        k: Number of documents to retrieve per query.
    """
    docs = _load_and_split_docs(spec)
    return BM25Retriever.from_documents(docs, k=k)
