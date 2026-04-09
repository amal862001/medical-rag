from collections import defaultdict
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import Field


def normalize_retrieved_documents(docs: List[Document]) -> List[Document]:
    normalized_docs = []

    for doc in docs:
        meta = doc.metadata or {}
        content_type = (
            meta.get("content_type")
            or meta.get("chunk_type")
            or "text"
        )

        normalized_meta = {
            "content_type": content_type,
            "source_pdf": meta.get("source_pdf") or meta.get("source") or "unknown",
            "page": meta.get("page", "unknown"),
            "chunk_id": meta.get("chunk_id", "unknown"),
            "retrieval_type": meta.get("retrieval_type", "vector_search"),
        }

        if content_type == "table":
            normalized_meta.update({
                "table_id": meta.get("table_id", "unknown"),
                "table_chunk_index": meta.get("table_chunk_index", "unknown"),
            })
            doc.page_content = (
                f"[TABLE DATA]\n"
                f"Table ID: {meta.get('table_id', 'unknown')}\n\n"
                f"{doc.page_content}"
            )

        if content_type == "image":
            normalized_meta.update({
                "image_id": meta.get("image_id", "unknown"),
                "image_path": meta.get("image_path", "unknown"),
            })
            doc.page_content = f"[IMAGE DESCRIPTION]\n{doc.page_content}"

        doc.metadata = normalized_meta
        normalized_docs.append(doc)

    return normalized_docs


def diversify_documents(
    docs: List[Document],
    max_docs: int = 12,
    max_per_source: int = 3,
) -> List[Document]:
    chosen = []
    per_source = defaultdict(int)
    seen = set()

    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source_pdf") or meta.get("source") or "unknown"
        page = meta.get("page", "unknown")
        chunk_id = meta.get("chunk_id", "unknown")
        dedupe_key = (source, page, chunk_id)

        if dedupe_key in seen:
            continue
        if per_source[source] >= max_per_source:
            continue

        seen.add(dedupe_key)
        per_source[source] += 1
        chosen.append(doc)

        if len(chosen) >= max_docs:
            break

    return chosen


class MultiModalChunkRetriever(BaseRetriever):
    vectorstore: Chroma = Field(...)
    top_k: int = Field(default=12)
    fetch_k: int = Field(default=36)
    max_per_source: int = Field(default=3)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            docs = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=self.top_k,
                fetch_k=self.fetch_k,
            )
            retrieval_type = "mmr_search"
        except Exception:
            docs = self.vectorstore.similarity_search(
                query=query,
                k=self.fetch_k,
            )
            retrieval_type = "similarity_search"

        docs = normalize_retrieved_documents(docs)
        for doc in docs:
            doc.metadata["retrieval_type"] = retrieval_type

        return diversify_documents(
            docs,
            max_docs=self.top_k,
            max_per_source=self.max_per_source,
        )


def get_multimodal_retriever(
    chroma_dir: str,
    collection_name: str = "chunk_store",
    top_k: int = 12,
    fetch_k: int = 36,
    max_per_source: int = 3,
) -> MultiModalChunkRetriever:
    embedding_fn = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    vectorstore = Chroma(
        persist_directory=chroma_dir,
        collection_name=collection_name,
        embedding_function=embedding_fn,
    )

    return MultiModalChunkRetriever(
        vectorstore=vectorstore,
        top_k=top_k,
        fetch_k=fetch_k,
        max_per_source=max_per_source,
    )
