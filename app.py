import os
import sys
import subprocess
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
OCR_DIR = DATA_DIR / "_ocr_text"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "my_notes"
EMBED_MODEL = "nomic-embed-text"

SUPPORTED_EXTENSIONS = [
    ".pdf", ".md", ".txt", ".docx", ".pptx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"
]
ocr_search_query = ""
show_ocr_preview = False

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
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )


def rebuild_index():
    DATA_DIR.mkdir(exist_ok=True)
    OCR_DIR.mkdir(exist_ok=True)

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

def get_ocr_files():
    if not OCR_DIR.exists():
        return []

    return sorted(OCR_DIR.glob("*.txt"))


def get_ocr_stats():
    ocr_files = get_ocr_files()

    total_characters = 0

    for file in ocr_files:
        try:
            total_characters += len(file.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass

    return {
        "ocr_folder_exists": OCR_DIR.exists(),
        "file_count": len(ocr_files),
        "total_characters": total_characters,
    }


def search_ocr_text(query: str):
    query = query.strip().lower()

    if not query:
        return []

    results = []

    for file in get_ocr_files():
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        lower_text = text.lower()

        if query in lower_text:
            index = lower_text.find(query)
            start = max(0, index - 300)
            end = min(len(text), index + 700)

            results.append(
                {
                    "file": file.name,
                    "preview": text[start:end],
                }
            )

    return results


def run_ocr_script():
    script_path = Path("make_ocr_notes.py")

    if not script_path.exists():
        return False, "make_ocr_notes.py was not found in your project folder."

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            cwd=Path.cwd(),
            timeout=600,
        )

        output = result.stdout + "\n" + result.stderr

        if result.returncode != 0:
            return False, output

        return True, output

    except subprocess.TimeoutExpired:
        return False, "OCR took too long and timed out. Try uploading a smaller PDF."

    except Exception as e:
        return False, f"OCR failed: {e}"
    

with st.sidebar:
    st.header("Settings")

    llm_model = st.selectbox(
        "Choose local model",
        ["qwen2.5:0.5b"],
        index=0,
        key="llm_model_select",
    )

    top_k = st.slider(
        "Number of note chunks to retrieve",
        min_value=1,
        max_value=5,
        value=2,
        key="top_k_slider",
    )

    context_window = st.selectbox(
        "Context window",
        [1024, 2048],
        index=0,
        key="context_window_select",
    )

    debug_mode = st.checkbox(
        "Show retrieval debug info",
        value=False,
        key="retrieval_debug_checkbox",
    )

    show_full_chunks = st.checkbox(
        "Show full retrieved chunks",
        value=False,
        key="full_chunks_checkbox",
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload notes",
        type=[
            "pdf", "md", "txt", "docx", "pptx",
            "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"
        ],
        accept_multiple_files=True,
        key="notes_uploader",
    )

    if st.button("Save uploaded files", key="save_files_button"):
        DATA_DIR.mkdir(exist_ok=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = DATA_DIR / uploaded_file.name
                file_path.write_bytes(uploaded_file.getbuffer())

            st.success(f"Saved {len(uploaded_files)} file(s) to the data folder.")
        else:
            st.warning("Please upload at least one file first.")

    if st.button("Re-index notes", key="reindex_button"):
        with st.spinner("Indexing your notes... This may take time on Hugging Face free CPU."):
            success = rebuild_index()

        if success:
            st.success("Index rebuilt successfully.")

    st.divider()
    st.subheader("OCR Tools")

    ocr_stats = get_ocr_stats()

    if ocr_stats["ocr_folder_exists"]:
        st.caption("OCR folder found.")
    else:
        st.caption("OCR folder not found yet.")

    st.write(f"OCR text files: {ocr_stats['file_count']}")
    st.write(f"OCR text characters: {ocr_stats['total_characters']}")

    if st.button("Generate OCR text files", key="generate_ocr_button"):
        with st.spinner("Running OCR on uploaded PDFs/images... This may take time on Hugging Face free CPU."):
            success, output = run_ocr_script()

        if success:
            st.success("OCR text files generated.")
        else:
            st.error("OCR generation failed.")

        with st.expander("OCR script output"):
            st.text(output)

    ocr_search_query = st.text_input(
        "Search OCR text",
        placeholder="Example: Swing, MVC, platform",
        key="ocr_search_input",
    )

    show_ocr_preview = st.checkbox(
        "Show OCR preview",
        value=False,
        key="show_ocr_preview_checkbox",
    )

    st.divider()

    if st.button("Clear chat", key="clear_chat_button"):
        st.session_state.messages = []
        st.rerun()

setup_models(llm_model, context_window)

if show_ocr_preview:
    st.subheader("OCR Text Preview")

    ocr_files = get_ocr_files()

    if not ocr_files:
        st.warning("No OCR text files found. Click 'Generate OCR text files' in the sidebar first.")
    else:
        if ocr_search_query:
            results = search_ocr_text(ocr_search_query)

            st.write(f"Search results for: `{ocr_search_query}`")

            if not results:
                st.warning("No matches found in OCR text.")
            else:
                for result in results[:10]:
                    with st.expander(result["file"]):
                        st.text(result["preview"])
        else:
            selected_file = st.selectbox(
                "Choose an OCR text file to preview",
                ocr_files,
                format_func=lambda file: file.name,
            )

            text = selected_file.read_text(encoding="utf-8", errors="ignore")

            st.write(f"Previewing: `{selected_file.name}`")
            st.text(text[:5000])
    
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
                    chunk_text = node.node.get_content()
                    metadata = node.node.metadata

                    file_name = (
                        metadata.get("file_name")
                        or metadata.get("file_path")
                        or "Unknown file"
                    )

                    page = metadata.get("page_label") or metadata.get("page") or ""
                    source_type = metadata.get("source_type", "unknown")
                    score = getattr(node, "score", None)

                    source_label = f"{file_name}"
                    if page:
                        source_label += f" — page {page}"

                    context_parts.append(
                        f"Source {i}: {source_label}\n"
                        f"Source type: {source_type}\n"
                        f"Text:\n{chunk_text}"
                    )

                    sources.append(
                        {
                            "number": i,
                            "file_name": file_name,
                            "page": page,
                            "source_type": source_type,
                            "score": score,
                            "text": chunk_text,
                        }
                    )

                context = "\n\n---\n\n".join(context_parts)

                prompt = f"""
            You are a strict notes-based assistant.

            Answer the user's question using ONLY the context below.

            Rules:
            1. Do not use outside knowledge.
            2. Do not guess.
            3. If the answer is clearly present in the context, answer directly.
            4. If the answer is not clearly present in the context, say: "I could not find this in your notes."
            5. Mention the source file and page when possible.

            Context:
            {context}

            User question:
            {question}

            Answer:
            """

                answer = Settings.llm.complete(prompt).text.strip()

                st.markdown(answer)

                with st.expander("Sources used"):
                    for source in sources:
                        source_line = f"{source['number']}. {source['file_name']}"

                        if source["page"]:
                            source_line += f" — page {source['page']}"

                        if source["score"] is not None:
                            source_line += f" — score: {source['score']:.4f}"

                        st.write(source_line)

                if debug_mode:
                    st.divider()
                    st.subheader("Retrieval Debug Info")

                    if not sources:
                        st.warning("No chunks were retrieved from ChromaDB.")

                    for source in sources:
                        score_text = (
                            f"{source['score']:.4f}"
                            if source["score"] is not None
                            else "No score"
                        )

                        with st.expander(
                            f"Chunk {source['number']} | {source['file_name']} | Page {source['page']} | Score: {score_text}"
                        ):
                            st.write("**File:**", source["file_name"])
                            st.write("**Page:**", source["page"] or "Not available")
                            st.write("**Source type:**", source["source_type"])
                            st.write("**Score:**", score_text)

                            st.write("**Retrieved text preview:**")

                            if show_full_chunks:
                                st.text(source["text"])
                            else:
                                st.text(source["text"][:1200])
            except Exception as e:
                answer = f"Error: {e}"
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})