import tempfile

import streamlit as st
from langchain_core.messages import HumanMessage


from bootstrap import (
    ensure_biomedical_vectorstore,
    get_embedding_function,
    prepare_documents_for_indexing,
    get_vectorstore,
)
from config import CHROMA_DIR
from pdf_loader import load_pdf_documents
from rag_chain import QA_PROMPT, SUMMARY_PROMPT, detect_intent, get_llm, rag_qa


st.set_page_config(page_title="Biomedical RAG Assistant", layout="wide")
st.title("Biomedical Research Assistant")


@st.cache_resource
def get_cached_embedding_function():
    return get_embedding_function()


@st.cache_resource
def get_cached_biomedical_store():
    return ensure_biomedical_vectorstore()


embedding_fn = get_cached_embedding_function()

if "history_bio" not in st.session_state:
    st.session_state.history_bio = []

if "history_upload" not in st.session_state:
    st.session_state.history_upload = []

if "upload_store" not in st.session_state:
    st.session_state.upload_store = None


mode = st.sidebar.radio(
    "Select Mode",
    ["Biomedical Literature Q&A", "Uploaded Document Q&A"],
)


if mode == "Biomedical Literature Q&A":
    st.header("Biomedical Literature Q&A")

    try:
        with st.spinner("Preparing biomedical knowledge base..."):
            _, store_status = get_cached_biomedical_store()
        if store_status == "built":
            st.info(f"Built a new local vector database from PDFs in {CHROMA_DIR}.")
    except Exception as exc:
        st.error(f"Biomedical knowledge base is not ready: {exc}")
        st.stop()

    query = st.chat_input("Ask a biomedical question or request a summary")

    if query:
        with st.spinner("Retrieving answer..."):
            answer, docs, _ = rag_qa(query, k=6)

        st.session_state.history_bio.append((query, answer))

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Evidence")
        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata or {}
            content_type = meta.get("content_type") or meta.get("chunk_type", "text")
            pdf = meta.get("source_pdf") or meta.get("source", "Unknown")
            st.markdown(f"**{i}. PDF:** {pdf} | **Type:** {content_type}")
            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

        st.subheader("History")
        for saved_query, saved_answer in reversed(st.session_state.history_bio):
            st.markdown(f"**Q:** {saved_query}")
            st.markdown(f"**A:** {saved_answer}")
            st.markdown("---")

else:
    st.header("Uploaded Document Q&A")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        with st.spinner("Indexing uploaded documents..."):
            st.session_state.upload_store = get_vectorstore(
                persist_directory=None,
                embedding_function=embedding_fn,
            )

            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    temp_path = tmp.name

                docs = load_pdf_documents(temp_path, file.name)
                docs = prepare_documents_for_indexing(docs)
                st.session_state.upload_store.add_documents(docs)

        st.success("Uploaded PDFs indexed successfully")

    query = st.chat_input("Ask a question about uploaded PDFs")

    if query:
        if not st.session_state.upload_store:
            st.warning("Please upload documents first")
        else:
            docs = st.session_state.upload_store.similarity_search(query, k=6)

            context_blocks = []
            for doc in docs:
                meta = doc.metadata or {}
                pdf = meta.get("source_pdf") or meta.get("source", "Unknown")
                context_blocks.append(f"Source: {pdf}\n{doc.page_content}")

            context = "\n\n---\n\n".join(context_blocks)

            intent = detect_intent(query)
            prompt = (
                SUMMARY_PROMPT.format(context=context, question=query)
                if intent == "summary"
                else QA_PROMPT.format(context=context, question=query)
            )

            response = get_llm().generate([[HumanMessage(content=prompt)]])
            answer = response.generations[0][0].text.strip()

            st.session_state.history_upload.append((query, answer))

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Evidence")
            for i, doc in enumerate(docs, start=1):
                meta = doc.metadata or {}
                pdf = meta.get("source_pdf") or meta.get("source", "Unknown")
                st.markdown(f"**{i}. PDF:** {pdf}")
                st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

            st.subheader("History")
            for saved_query, saved_answer in reversed(st.session_state.history_upload):
                st.markdown(f"**Q:** {saved_query}")
                st.markdown(f"**A:** {saved_answer}")
                st.markdown("---")
