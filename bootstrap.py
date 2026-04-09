from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    SOURCE_PDF_DIR,
)
from pdf_loader import load_pdf_documents


ProgressCallback = Callable[[str], None]
MAX_TABLE_CHARS = 1200


def get_embedding_function() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )


def get_vectorstore(
    persist_directory: str | Path | None = None,
    embedding_function: HuggingFaceEmbeddings | None = None,
) -> Chroma:
    persist_directory = Path(persist_directory) if persist_directory else None
    if persist_directory:
        persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function or get_embedding_function(),
        persist_directory=str(persist_directory) if persist_directory else None,
    )


def _chunk_table_document(doc: Document) -> list[Document]:
    lines = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
    if not lines:
        return []

    chunks = []
    buffer = ""
    chunk_index = 0

    for line in lines:
        candidate = f"{buffer}\n{line}".strip() if buffer else line
        if len(candidate) <= MAX_TABLE_CHARS:
            buffer = candidate
            continue

        metadata = deepcopy(doc.metadata)
        metadata["table_chunk_index"] = chunk_index
        chunks.append(Document(page_content=buffer, metadata=metadata))
        buffer = line
        chunk_index += 1

    if buffer:
        metadata = deepcopy(doc.metadata)
        metadata["table_chunk_index"] = chunk_index
        chunks.append(Document(page_content=buffer, metadata=metadata))

    return chunks


def prepare_documents_for_indexing(documents: list[Document]) -> list[Document]:
    text_splitter = get_text_splitter()
    prepared = []

    for doc in documents:
        metadata = deepcopy(doc.metadata or {})
        content_type = metadata.get("content_type", "text")
        source = metadata.get("source_pdf") or metadata.get("source", "unknown")
        metadata["source"] = source
        metadata["source_pdf"] = source

        base_doc = Document(page_content=doc.page_content, metadata=metadata)

        if content_type == "text":
            prepared.extend(text_splitter.split_documents([base_doc]))
        elif content_type == "table":
            prepared.extend(_chunk_table_document(base_doc))
        else:
            prepared.append(base_doc)

    for index, doc in enumerate(prepared):
        doc.metadata["chunk_id"] = index

    return prepared


def chroma_has_documents(chroma_dir: str | Path = CHROMA_DIR) -> bool:
    chroma_dir = Path(chroma_dir)
    if not chroma_dir.exists():
        return False

    store = get_vectorstore(chroma_dir)
    return store._collection.count() > 0


def _iter_source_pdfs(pdf_dir: str | Path = SOURCE_PDF_DIR) -> list[Path]:
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        return []
    return sorted(pdf_dir.glob("*.pdf"))


def build_chroma_from_raw_pdfs(
    raw_pdfs_dir: str | Path = SOURCE_PDF_DIR,
    chroma_dir: str | Path = CHROMA_DIR,
    progress_callback: ProgressCallback | None = None,
) -> Chroma:
    pdf_paths = _iter_source_pdfs(raw_pdfs_dir)
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{Path(raw_pdfs_dir)}'. "
            "Add PDF files inside data/raw_pdfs so the vector database can be built automatically."
        )

    def report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    embedding_fn = get_embedding_function()
    vectorstore = get_vectorstore(chroma_dir, embedding_fn)

    if vectorstore._collection.count() > 0:
        report("Existing Chroma collection already has vectors.")
        return vectorstore

    report(f"Building Chroma DB from {len(pdf_paths)} PDF files...")

    running_index = 0
    for pdf_path in pdf_paths:
        report(f"Indexing {pdf_path.name}...")
        docs = load_pdf_documents(str(pdf_path), pdf_path.name)
        indexed_docs = prepare_documents_for_indexing(docs)

        ids = []
        for doc in indexed_docs:
            source = doc.metadata.get("source", pdf_path.name)
            page = doc.metadata.get("page", "na")
            ids.append(f"{source}-p{page}-c{running_index}")
            running_index += 1

        if indexed_docs:
            vectorstore.add_documents(indexed_docs, ids=ids)

    report(f"Chroma DB ready with {vectorstore._collection.count()} chunks.")
    return vectorstore


def ensure_biomedical_vectorstore(
    progress_callback: ProgressCallback | None = None,
) -> tuple[Chroma, str]:
    if chroma_has_documents(CHROMA_DIR):
        return get_vectorstore(CHROMA_DIR), "loaded"

    return (
        build_chroma_from_raw_pdfs(
            raw_pdfs_dir=SOURCE_PDF_DIR,
            chroma_dir=CHROMA_DIR,
            progress_callback=progress_callback,
        ),
        "built",
    )
