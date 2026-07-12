"""Document ingestion pipeline for DocWhisperer.

Handles loading, chunking, embedding, and storing documents in ChromaDB,
plus building and persisting a BM25 index for lexical search.
"""
import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from backend.config import Settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


@dataclass
class IngestionResult:
    """Summary of a document ingestion run.

    Attributes:
        files_processed: Number of files successfully processed.
        chunks_created: Total number of chunks stored.
        errors: List of error messages for files that failed.
    """

    files_processed: int = 0
    chunks_created: int = 0
    errors: List[str] = field(default_factory=list)


class DocumentIngester:
    """Ingests documents into ChromaDB with BM25 index support.

    Args:
        settings: Application settings with DB paths and API keys.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            length_function=len,
        )
        self._embedder = self._create_embedder()
        self._vectorstore = self._create_vectorstore()
        self._all_chunks: List[Dict[str, Any]] = []

    def _create_embedder(self) -> Any:
        """Create the embedding model instance.

        Returns:
            An embeddings instance (OpenAI or a fallback).
        """
        if self._settings.openai_api_key:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=self._settings.openai_api_key,
            )
        else:
            from langchain_community.embeddings import OllamaEmbeddings

            return OllamaEmbeddings(
                base_url=self._settings.ollama_url or "http://localhost:11434",
                model="nomic-embed-text",
            )

    def _create_vectorstore(self) -> Any:
        """Create or connect to the ChromaDB vector store.

        Returns:
            A LangChain Chroma vectorstore instance.
        """
        from langchain_community.vectorstores import Chroma

        return Chroma(
            collection_name="docwhisperer",
            embedding_function=self._embedder,
            persist_directory=self._settings.chroma_db_dir,
        )

    @property
    def collection(self) -> Any:
        """Return the underlying ChromaDB collection."""
        return self._vectorstore._collection

    @property
    def embedder(self) -> Any:
        """Return the embedder instance."""
        return self._embedder

    def ingest_directory(self, path: str) -> IngestionResult:
        """Ingest all supported documents from a directory.

        Args:
            path: Path to the directory containing documents.

        Returns:
            IngestionResult with processing statistics.
        """
        result = IngestionResult()
        dir_path = Path(path)

        if not dir_path.is_dir():
            result.errors.append(f"Directory not found: {path}")
            return result

        files = [
            f for f in dir_path.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        logger.info("Found %d supported files in %s", len(files), path)

        for file_path in files:
            try:
                chunks = self.ingest_file(str(file_path))
                self._embed_and_store(chunks)
                result.files_processed += 1
                result.chunks_created += len(chunks)
                logger.info("Ingested %s → %d chunks", file_path.name, len(chunks))
            except Exception as exc:
                msg = f"Failed to ingest {file_path}: {exc}"
                logger.error(msg)
                result.errors.append(msg)

        return result

    def ingest_file(self, file_path: str) -> List[Document]:
        """Load and chunk a single file.

        Args:
            file_path: Path to the file.

        Returns:
            List of chunked Document instances.
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
        elif ext in (".md", ".markdown"):
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            loader = UnstructuredMarkdownLoader(file_path)
        elif ext == ".txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        raw_docs = loader.load()
        chunks: List[Document] = []
        for doc in raw_docs:
            doc.metadata.setdefault("source", file_path)
            chunks.extend(self._chunk_document(doc))
        return chunks

    def _chunk_document(self, doc: Document) -> List[Document]:
        """Split a document into chunks.

        Args:
            doc: A LangChain Document to split.

        Returns:
            List of chunked Documents.
        """
        return self._splitter.split_documents([doc])

    def _embed_and_store(self, chunks: List[Document]) -> None:
        """Embed chunks and store them in ChromaDB.

        Args:
            chunks: List of Document chunks to embed and store.
        """
        if not chunks:
            return

        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]

        self._vectorstore.add_texts(texts=texts, metadatas=metadatas)

        for i, chunk in enumerate(chunks):
            self._all_chunks.append({
                "id": f"chunk_{len(self._all_chunks)}",
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            })

    def _build_bm25_index(self) -> BM25Okapi:
        """Build a BM25 index from all ingested chunks.

        Returns:
            A BM25Okapi index instance.
        """
        if not self._all_chunks:
            raise ValueError("No chunks available to build BM25 index")

        tokenized_corpus = [doc["text"].lower().split() for doc in self._all_chunks]
        return BM25Okapi(tokenized_corpus)

    def save_bm25_index(self, path: str) -> None:
        """Build and save the BM25 index and corpus docs to disk.

        Args:
            path: File path to save the pickled index data.
        """
        bm25 = self._build_bm25_index()
        data = {"bm25": bm25, "corpus_docs": self._all_chunks}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("BM25 index saved to %s (%d docs)", path, len(self._all_chunks))

    def load_bm25_index(self, path: str) -> BM25Okapi:
        """Load a previously saved BM25 index from disk.

        Args:
            path: File path to the pickled index data.

        Returns:
            A BM25Okapi index instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._all_chunks = data.get("corpus_docs", [])
        logger.info("BM25 index loaded from %s (%d docs)", path, len(self._all_chunks))
        return data["bm25"]

    def get_corpus_docs(self) -> List[Dict[str, Any]]:
        """Return all ingested chunk metadata for BM25 mapping.

        Returns:
            List of dicts with id, text, metadata.
        """
        return self._all_chunks
