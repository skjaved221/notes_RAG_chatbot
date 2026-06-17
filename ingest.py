import chromadb

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from ocr_loader import load_notes_with_ocr

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "my_notes"

LLM_MODEL = "llama3.2:1b"
EMBED_MODEL = "nomic-embed-text"

Settings.llm = Ollama(
    model=LLM_MODEL,
    request_timeout=300.0,
    context_window=1024
)

Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL,
    base_url="http://localhost:11434"
)

Settings.node_parser = SentenceSplitter(
    chunk_size=800,
    chunk_overlap=120
)

print("Loading documents...")

documents = load_notes_with_ocr(DATA_DIR)
if not documents:
    raise ValueError("No supported files found in the data folder.")

print(f"Loaded {len(documents)} documents.")

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    chroma_client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Creating vector index with local Ollama embeddings...")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

print("Done. Your notes are indexed locally.")