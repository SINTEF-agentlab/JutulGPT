import os
from contextlib import contextmanager
from typing import Generator, TypedDict

from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever

from jutulgpt.configuration import BaseConfiguration
from jutulgpt.rag.retriever_specs import RetrieverSpec
from jutulgpt.utils.model import get_provider_and_model


class RetrievalParams(TypedDict):
    search_type: str
    search_kwargs: dict


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split(":", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=model)

        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


def _load_and_split_docs(spec: RetrieverSpec) -> list:
    """Load documents from *spec.dir_path*, split them, and return chunks.

    If *spec.cache_path* is set and the cache file exists, the raw documents
    are loaded from the pickle cache.  Otherwise they are loaded from disk via
    :class:`DirectoryLoader` (and optionally cached for next time).
    """
    import pickle

    from langchain_community.document_loaders import DirectoryLoader, TextLoader

    if isinstance(spec.filetype, str):
        filetypes = [spec.filetype]
    else:
        filetypes = spec.filetype

    # Try loading from cache first (only when a cache_path is configured)
    if spec.cache_path and os.path.exists(spec.cache_path):
        with open(spec.cache_path, "rb") as f:
            docs = pickle.load(f)
    else:
        loaders = []
        for filetype in filetypes:
            loader = DirectoryLoader(
                path=spec.dir_path,
                glob=f"**/*.{filetype}",
                show_progress=False,
                loader_cls=TextLoader,
            )
            loaders.append(loader)

        docs = []
        for loader in loaders:
            docs.extend(loader.load())

        # Persist to cache if a cache_path was given
        if spec.cache_path:
            os.makedirs(os.path.dirname(spec.cache_path), exist_ok=True)
            with open(spec.cache_path, "wb") as f:
                pickle.dump(docs, f)

    # Split documents
    chunks = []
    for doc in docs:
        chunks.extend(spec.split_func(doc))
    return chunks


# ---------------------------------------------------------------------------
# BM25 retriever (no embeddings, no persistence)
# ---------------------------------------------------------------------------


@contextmanager
def make_bm25_retriever(
    spec: RetrieverSpec,
    search_kwargs: dict,
) -> Generator:
    """Create a BM25 retriever from the documents in *spec*.

    Documents are loaded and split on every call (fast for typical doc sizes).
    No embeddings or vector store are required.
    """
    from langchain_community.retrievers import BM25Retriever

    docs = _load_and_split_docs(spec)
    k = search_kwargs.get("k", 3)
    retriever = BM25Retriever.from_documents(docs, k=k)
    yield retriever


# ---------------------------------------------------------------------------
# Vector-store retrievers (FAISS / Chroma) – kept for backward-compatibility
# ---------------------------------------------------------------------------


@contextmanager
def make_faiss_retriever(
    configuration: BaseConfiguration,
    spec: RetrieverSpec,
    embedding_model: Embeddings,
    search_type: str,
    search_kwargs: dict,
) -> Generator[VectorStoreRetriever, None, None]:
    """
    Create or load a FAISS retriever, saving the index locally to avoid re-indexing.
    Uses configuration to determine file paths and splitting functions.
    """
    import os

    from langchain_community.vectorstores import FAISS

    # Get the persist path by checking what is the specified embedding model
    persist_path = spec.persist_path(
        get_provider_and_model(configuration.embedding_model)[0]
    )

    # Load or create FAISS index
    if os.path.exists(persist_path):
        vectorstore = FAISS.load_local(
            persist_path,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
    else:
        print(f"Creating new FAISS index at {spec.persist_path}")
        docs = _load_and_split_docs(spec)
        vectorstore = FAISS.from_documents(
            documents=docs,
            embedding=embedding_model,
        )
        vectorstore.save_local(persist_path)

    yield vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={**search_kwargs},
    )


@contextmanager
def make_chroma_retriever(
    configuration: BaseConfiguration,
    spec: RetrieverSpec,
    embedding_model: Embeddings,
    search_type: str,
    search_kwargs: dict,
) -> Generator[VectorStoreRetriever, None, None]:
    """
    Create or load a Chroma retriever, saving the index locally to avoid re-indexing.
    Uses configuration to determine file paths and splitting functions.
    """
    import os

    from langchain_chroma import Chroma

    # Get the persist path by checking what is the specified embedding model
    persist_path = spec.persist_path(
        get_provider_and_model(configuration.embedding_model)[0]
    )

    # Load or create Chroma index
    if os.path.exists(persist_path):
        vectorstore = Chroma(
            embedding_function=embedding_model,
            persist_directory=persist_path,
            collection_name=spec.collection_name,
        )

    else:
        print(f"Creating new Chroma index at {persist_path}")
        docs = _load_and_split_docs(spec)

        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=persist_path,
            collection_name=spec.collection_name,
        )

    yield vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={**search_kwargs},
    )


@contextmanager
def make_retriever(
    config: RunnableConfig,
    spec: RetrieverSpec,
    retrieval_params: RetrievalParams = RetrievalParams(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 15, "lambda_mult": 0.5},
    ),
) -> Generator:
    """
    Create a retriever for the agent, based on the current configuration.

    Args:
        config: The runnable configuration
        spec: The retriever specification
        retrieval_params: Override retrieval parameters (search_type, search_kwargs, etc.)
    """
    configuration = BaseConfiguration.from_runnable_config(config)

    # Get the retriever
    selected_retriever = None
    match configuration.retriever_provider:
        case "bm25":
            with make_bm25_retriever(
                spec,
                retrieval_params["search_kwargs"],
            ) as retriever:
                selected_retriever = retriever

        case "faiss":
            embedding_model = make_text_encoder(configuration.embedding_model)
            with make_faiss_retriever(
                configuration,
                spec,
                embedding_model,
                retrieval_params["search_type"],
                retrieval_params["search_kwargs"],
            ) as retriever:
                selected_retriever = retriever
        case "chroma":
            embedding_model = make_text_encoder(configuration.embedding_model)
            with make_chroma_retriever(
                configuration,
                spec,
                embedding_model,
                retrieval_params["search_type"],
                retrieval_params["search_kwargs"],
            ) as retriever:
                selected_retriever = retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )

    # Apply the reranker
    match configuration.rerank_provider:
        case "None":
            yield selected_retriever
        case _:
            raise ValueError(
                "Unrecognized rerank_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['rerank_provider'].__args__)}\n"
                f"Got: {configuration.rerank_provider}"
            )
