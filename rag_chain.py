import os
import re
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from bootstrap import ensure_biomedical_vectorstore
from similarity_search import diversify_documents, normalize_retrieved_documents


DEFAULT_K = 12
FETCH_K_MULTIPLIER = 4


QA_PROMPT = """
You are a biomedical research assistant.

Answer the question ONLY using the provided context.
Do not add external knowledge.
If the answer is not present, say:
"I could not find this information in the provided documents."

When possible, cite evidence inline using [Source, p.X].

Context:
{context}

Question:
{question}

Answer in a concise academic tone:
"""


SUMMARY_PROMPT = """
You are a biomedical research assistant.

Create a focused summary using ONLY the provided context.
Do not add external knowledge.
If the context is insufficient, say so clearly.

When possible, cite evidence inline using [Source, p.X].

Context:
{context}

User request:
{question}

Summary:
"""


def detect_intent(query: str) -> str:
    lowered = query.lower()
    summary_keywords = ("summarize", "summary", "overview", "brief", "key findings")
    return "summary" if any(keyword in lowered for keyword in summary_keywords) else "qa"


def clean_query(query: str) -> str:
    cleaned = re.sub(r"\bpdf only\b", " ", query, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpdfs only\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpdf\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY or GOOGLE_API_KEY. "
            "Set it in the environment or .env before running the app."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=1024,
        google_api_key=api_key,
    )


llm = None


def format_context(docs: List[Document]) -> str:
    blocks = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source_pdf") or metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        blocks.append(f"Source: {source} | Page: {page}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def format_citations(docs: List[Document]) -> str:
    seen = []
    seen_keys = set()

    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source_pdf") or meta.get("source", "unknown")
        page = meta.get("page", "?")
        key = (source, page)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        seen.append(f"- {source}, p.{page}")

    return "\n".join(seen)


def retrieve_documents(query: str, k: int = DEFAULT_K) -> List[Document]:
    vectorstore, _ = ensure_biomedical_vectorstore()
    cleaned_query = clean_query(query)
    fetch_k = max(k * FETCH_K_MULTIPLIER, 24)

    try:
        docs = vectorstore.max_marginal_relevance_search(
            cleaned_query,
            k=k,
            fetch_k=fetch_k,
        )
    except Exception:
        docs = vectorstore.similarity_search(cleaned_query, k=fetch_k)

    docs = normalize_retrieved_documents(docs)
    return diversify_documents(docs, max_docs=k, max_per_source=3)


def rag_qa(query: str, k: int = DEFAULT_K) -> tuple[str, List[Document], str]:
    docs = retrieve_documents(query, k=k)

    if not docs:
        return "No relevant documents found.", [], "qa"

    context = format_context(docs)
    intent = detect_intent(query)
    prompt = (
        SUMMARY_PROMPT.format(context=context, question=query)
        if intent == "summary"
        else QA_PROMPT.format(context=context, question=query)
    )

    response = get_llm().generate([[HumanMessage(content=prompt)]])
    answer = response.generations[0][0].text.strip()
    citations = format_citations(docs)

    if citations:
        answer = f"{answer}\n\nReferences:\n{citations}"

    return answer, docs, intent
