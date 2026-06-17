import streamlit as st
import chromadb

from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "my_notes"

LLM_MODEL = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Chat with My Notes", page_icon="📚")
st.title("📚 Chat with My Notes — Free Local Version")


@st.cache_resource
def load_index():
    Settings.llm = Ollama(
        model=LLM_MODEL,
        request_timeout=300.0,
        context_window=1024
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url="http://localhost:11434"
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


index = load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="context",
        similarity_top_k=2,
        system_prompt=(
            "You are a helpful study assistant. Answer only using the provided notes. "
            "If the notes do not contain the answer, say you could not find it in the notes."
        )
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask something from your notes...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes locally..."):
            response = st.session_state.chat_engine.chat(question)
            answer = str(response)

            st.markdown(answer)

            if hasattr(response, "source_nodes") and response.source_nodes:
                with st.expander("Sources used"):
                    for i, source in enumerate(response.source_nodes, start=1):
                        metadata = source.node.metadata
                        file_name = metadata.get("file_name") or metadata.get("file_path") or "Unknown file"
                        page = metadata.get("page_label") or metadata.get("page") or ""

                        source_line = f"{i}. {file_name}"
                        if page:
                            source_line += f" — page {page}"

                        st.write(source_line)

    st.session_state.messages.append({"role": "assistant", "content": answer})