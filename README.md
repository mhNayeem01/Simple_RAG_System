# 🤖 Simple RAG System — Streamlit + ChromaDB  
**LLMs:** OpenAI · Ollama (local) · Gemini  
**Embeddings:** OpenAI · Nomic (Ollama) · Chroma Default · Gemini
<br>
<br>


<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue.svg"></a>
  <a href="#"><img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-app-red.svg"></a>
  <a href="#"><img alt="ChromaDB" src="https://img.shields.io/badge/Vector%20DB-ChromaDB-8A2BE2.svg"></a>
  <a href="#"><img alt="Ollama" src="https://img.shields.io/badge/Local%20LLM-Ollama-00A67E.svg"></a>
  <a href="#"><img alt="Gemini" src="https://img.shields.io/badge/Gemini-supported-6C63FF.svg"></a>
</p>
<br>

A minimal, production-friendly **RAG (Retrieval Augmented Generation)** app:
- Upload a PDF → it’s chunked and embedded into **ChromaDB**.
- Ask questions → app retrieves relevant chunks and answers with your selected **LLM**.
- Works with **OpenAI**, **Ollama** (local), and **Gemini** — with resilient timeouts + retries for Ollama and **manual** embeddings for Gemini.

---



## ✨ Features
- **Plug-and-play LLMs:** switch between **OpenAI**, **Ollama**, and **Gemini** from the sidebar.
- **Multiple embedding backends:** OpenAI · Nomic (via Ollama) · Chroma default · **Gemini (manual, robust parsing)**.
- **Resilient Ollama calls:** long timeouts, **3× retries with backoff**, and **keep_alive=30m** to avoid “Request timed out”.
- **Sentence-aware chunking** with overlap for better retrieval quality.
- **Per-embedding collections** in Chroma (`documents_<embedding>`), safe to switch models without conflicts.

---
<br>

## 🏗 Architecture

![Ollama](https://github.com/mhNayeem01/Simple_RAG_System/blob/main/chroma_db/img/diagram.png)

<br>


## 🖼 Project Screenshots

### Testing with Gemini llm and embedding model 
![Model Selection](https://github.com/mhNayeem01/Simple_RAG_System/blob/main/chroma_db/img/gemini.png)

<br>

### Testin with ollama llm and CromaDB embedding model 
![Ollama](https://github.com/mhNayeem01/Simple_RAG_System/blob/main/chroma_db/img/ollama.png)






<br>


# 🚀 Getting Started

## Prerequisites
- **Python 3.9+**
- **pip**
- **Ollama** (if you plan to run local models)

Ensure the server runs:
```bash
ollama serve
```
<br>


Pull  models:
```bash
ollama pull llama3.1:8b

ollama pull nomic-embed-text:latest
```

<br>

### Installation

```bash
# 1) (optional) create and activate a venv
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 2) install dependencies
pip install -r requirements.txt
```
<br>

### Environment Variables
Create a .env file next to rag_pdf.py:

```bash
# Only set what you need
OPENAI_API_KEY=sk-your-openai-key        # if using OpenAI
GEMINI_API_KEY=AIza-your-gemini-key      # if using Gemini
```
⚡ Note: You don’t need any key for Ollama.
<br>

### Run the App
Make sure Ollama is running:

```bash
ollama serve
```
<br>

Start the Streamlit app:
```bash
streamlit run rag_pdf.py
```
<br>

### Access the App

Streamlit opens at http://localhost:8501 by default.

---

✅ 


