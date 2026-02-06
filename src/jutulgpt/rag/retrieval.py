import os

from langchain_community.retrievers import BM25Retriever

from jutulgpt.rag.retriever_specs import RetrieverSpec


def _load_and_split_docs(spec: RetrieverSpec) -> list:
    import pickle

    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    # Resolve dir_path if it's a callable
    dir_path = spec.dir_path() if callable(spec.dir_path) else spec.dir_path

    # Load or cache documents
    if isinstance(spec.filetype, str):
        filetypes = [spec.filetype]
    else:
        filetypes = spec.filetype

    loaders = []
    for filetype in filetypes:
        loader = DirectoryLoader(
            path=dir_path,
            glob=f"**/*.{filetype}",
            show_progress=True,
            loader_cls=TextLoader,
        )
        loaders.append(loader)

    # Ensure cache directory exists
    cache_dir = os.path.dirname(spec.cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(spec.cache_path):
        with open(spec.cache_path, "rb") as f:
            docs = pickle.load(f)
    else:
        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        with open(spec.cache_path, "wb") as f:
            pickle.dump(docs, f)

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
