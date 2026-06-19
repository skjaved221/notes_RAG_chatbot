---

title: Notes RAG Bot
emoji: "📚"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 📚 Notes RAG Bot

An AI-powered chatbot that lets users upload notes, PDFs, images, and study material, then ask questions directly from those documents.

The app uses **OCR + RAG** to understand both normal text and text inside images/scanned PDFs.

🔗 **Live Demo:** https://huggingface.co/spaces/sheikhjaved/notes-rag-bot

---

## ✨ Features

* 📄 Upload PDFs, text files, Markdown notes, DOCX, PPTX, and images
* 🖼️ OCR support for scanned notes, screenshots, and image-based PDF slides
* 🔍 Retrieval-Augmented Generation using ChromaDB and LlamaIndex
* 🤖 Local LLM support using Ollama
* 📚 Ask questions directly from uploaded study material
* 🧠 Strict notes-based answering to reduce hallucination
* 🧾 Source display with file name, page number, and retrieval score
* 🛠️ Retrieval debug mode to inspect retrieved chunks
* 👁️ OCR preview and OCR text search
* 🌐 Deployed publicly on Hugging Face Spaces

---

## 🧠 How It Works

```text
User uploads notes/PDF/images
        ↓
Text is extracted using normal parsing + OCR
        ↓
Documents are split into chunks
        ↓
Chunks are converted into embeddings
        ↓
Embeddings are stored in ChromaDB
        ↓
User asks a question
        ↓
Relevant chunks are retrieved
        ↓
Ollama model answers using only retrieved notes
```

---

## 🛠️ Tech Stack

| Tool                | Purpose                                         |
| ------------------- | ----------------------------------------------- |
| Streamlit           | Web app interface                               |
| LlamaIndex          | Document loading, chunking, indexing, retrieval |
| ChromaDB            | Vector database                                 |
| Ollama              | Local LLM and embedding model runtime           |
| Tesseract OCR       | Extracting text from images/scanned PDFs        |
| PyMuPDF             | Reading and rendering PDF pages                 |
| Hugging Face Spaces | Public deployment                               |
| Docker              | Custom deployment environment                   |

---

## 📸 App Workflow

1. Upload notes or PDFs.
2. Click **Save uploaded files**.
3. Click **Generate OCR text files**.
4. Click **Re-index notes**.
5. Ask questions from your notes.
6. Use **OCR Preview** or **Retrieval Debug Info** to verify what the chatbot is reading.

---

## ✅ Example Questions

```text
What is Java Swing?
```

```text
What are the differences between AWT and Swing?
```

```text
What is event handling in Java?
```

```text
Explain JButton with example.
```

```text
What are the steps to compile and run an applet?
```

If the answer is not present in the uploaded notes, the chatbot is instructed to say:

```text
I could not find this in your notes.
```

---

## 🚀 Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/notes-rag-bot.git
cd notes-rag-bot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Download Ollama from:

```text
https://ollama.com
```

Then pull the required models:

```bash
ollama pull qwen2.5:0.5b
ollama pull nomic-embed-text
```

### 5. Install Tesseract OCR

For Windows, install Tesseract OCR and make sure this file exists:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

For Linux/Debian-based systems:

```bash
sudo apt-get install tesseract-ocr
```

### 6. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

## 📁 Project Structure

```text
notes-rag-bot/
│
├── app.py                # Main Streamlit app
├── ingest.py             # Indexing script
├── ocr_loader.py         # OCR-based document loader
├── make_ocr_notes.py     # Creates OCR text files
├── requirements.txt      # Python dependencies
├── Dockerfile            # Hugging Face Docker deployment
├── start.sh              # Startup script for Hugging Face
├── README.md             # Project documentation
│
├── data/                 # Uploaded notes and demo files
├── chroma_db/            # Local ChromaDB vector store
└── .streamlit/           # Streamlit config
```

---

## 🌐 Hugging Face Deployment

This project is deployed using **Hugging Face Spaces with Docker**.

The Docker setup installs:

* Python
* Streamlit
* Tesseract OCR
* Ollama
* Required Python dependencies

The app runs on port:

```text
7860
```

Important: Free Hugging Face Spaces run on limited CPU resources, so large PDFs or OCR-heavy files may take time.

---

## ⚠️ Limitations

* Free Hugging Face CPU can be slow with local LLMs.
* Uploaded files and ChromaDB may reset when the Space restarts.
* Large PDFs may take longer to OCR and index.
* OCR can make mistakes if the image quality is poor.
* The app currently uses a small local model for free deployment.

---

## 🔮 Future Improvements

* Hybrid search using vector search + keyword search
* Reranking for better retrieval accuracy
* Notes manager with delete/clear options
* Quiz and flashcard generation
* Better source citations
* Persistent cloud storage
* Optional hosted LLM API mode for faster responses

---

## 🙌 Built By

**Sk Javed**

A student project focused on building a practical AI study assistant using RAG, OCR, and local LLMs.

---

## ⭐ Support

If you like this project, consider giving it a star on GitHub.
