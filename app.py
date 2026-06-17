import os
from pathlib import Path

import chromadb
import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from ocr_loader import load_notes_with_ocr

DATA_DIR = Path("./data")
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "my_notes"
EMBED_MODEL = "nomic-embed-text"

SUPPORTED_EXTENSIONS = [
    ".pdf", ".md", ".txt", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
]


st.set_page_config(page_title="Chat with My Notes", page_icon="📚")
st.title("📚 Chat with My Notes")


def setup_models(llm_model: str, context_window: int):
    Settings.llm = Ollama(
        model=llm_model,
        request_timeout=300.0,
        context_window=context_window,
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url="http://localhost:11434",
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )


def rebuild_index():
    DATA_DIR.mkdir(exist_ok=True)

    documents = load_notes_with_ocr(DATA_DIR)

    if not documents:
        st.error("No supported files found in the data folder.")
        return False

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    return True


def load_index():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    if chroma_collection.count() == 0:
        return None

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


with st.sidebar:
    st.header("Settings")

    llm_model = st.selectbox(
        "Choose local model",
        [
            "llama3.2:1b",
            "qwen2.5:0.5b",
            "llama3.2:3b",
        ],
        index=0,
    )

    top_k = st.slider(
        "Number of note chunks to retrieve",
        min_value=1,
        max_value=5,
        value=2,
    )

    context_window = st.selectbox(
        "Context window",
        [1024, 2048, 4096],
        index=0,
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload notes",
        type=["pdf", "md", "txt", "docx", "pptx", "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if st.button("Save uploaded files"):
        DATA_DIR.mkdir(exist_ok=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = DATA_DIR / uploaded_file.name
                file_path.write_bytes(uploaded_file.getbuffer())

            st.success(f"Saved {len(uploaded_files)} file(s) to the data folder.")
        else:
            st.warning("Please upload at least one file first.")

    if st.button("Re-index notes"):
        with st.spinner("Indexing your notes..."):
            success = rebuild_index()

        if success:
            st.success("Index rebuilt successfully.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


setup_models(llm_model, context_window)

index = load_index()

if index is None:
    st.info("Upload notes or add files to the data folder, then click Re-index notes.")
    st.stop()


retriever = index.as_retriever(similarity_top_k=top_k)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something from your notes...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes locally..."):
            try:
                nodes = retriever.retrieve(question)

                context_parts = []
                sources = []

                for i, node in enumerate(nodes, start=1):
                    text = node.node.get_content()
                    metadata = node.node.metadata

                    file_name = (
                        metadata.get("file_name")
                        or metadata.get("file_path")
                        or "Unknown file"
                    )

                    page = metadata.get("page_label") or metadata.get("page") or ""

                    source_label = f"{file_name}"
                    if page:
                        source_label += f" — page {page}"

                    context_parts.append(f"Source {i}: {source_label}\n{text}")
                    sources.append(source_label)

                context = "\n\n---\n\n".join(context_parts)

                prompt = f"""
                You are a strict notes-based assistant.

                Answer the user's question using ONLY the context below.

                If the answer is not clearly present in the context, say:
                "I could not find this in your notes."

                Do not use outside knowledge.
                Do not guess.
                Do not answer from unrelated context.

                Context:
                {context}

                User question:
                {question}

                Answer:
                """

                answer = Settings.llm.complete(prompt).text

                st.markdown(answer)

                with st.expander("Sources used"):
                    for i, source in enumerate(sources, start=1):
                        st.write(f"{i}. {source}")
            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})