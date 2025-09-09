import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import PyPDF2
import uuid
import time

# ---- NEW: Gemini client ----
try:
    from google import genai as google_genai
except Exception:
    google_genai = None

# Load environment variables
load_dotenv()

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---- Ollama tuning knobs ----
OLLAMA_MODEL = "llama3.1:8b"   # change if you use another local model (e.g. llama3.2:3b)
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OLLAMA_TIMEOUT_S = 120         # generous timeout for cold loads
OLLAMA_RETRIES = 3             # retry on timeouts
OLLAMA_KEEP_ALIVE = "30m"      # keep model hot

def _ollama_options():
    return {
        "num_ctx": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        # "num_predict": 768,  # uncomment to cap tokens (shorter = faster)
    }


class SimpleModelSelector:
    """Simple class to handle model selection"""

    def __init__(self):
        # Available LLM models
        self.llm_models = {
            "openai": "GPT-4o mini",
            "ollama": "Llama3.1 8B (local)",
            "gemini": "Gemini 2.5 Flash",
        }

        # Available embedding models with their dimensions
        self.embedding_models = {
            "openai": {
                "name": "OpenAI Embeddings",
                "dimensions": 1536,
                "model_name": "text-embedding-3-small",
            },
            "chroma": {"name": "Chroma Default", "dimensions": 384, "model_name": None},
            "nomic": {
                "name": "Nomic Embed Text (Ollama)",
                "dimensions": 768,
                "model_name": "nomic-embed-text",
            },
            # ---- NEW: Gemini embeddings
            "gemini": {
                "name": "Gemini Embeddings (text-embedding-004)",
                "dimensions": 768,
                "model_name": "text-embedding-004",
            },
        }

    def select_models(self):
        """Let user select models through Streamlit UI"""
        st.sidebar.title("📚 Model Selection")

        # Select LLM
        llm = st.sidebar.radio(
            "Choose LLM Model:",
            options=list(self.llm_models.keys()),
            format_func=lambda x: self.llm_models[x],
        )

        # Select Embeddings
        embedding = st.sidebar.radio(
            "Choose Embedding Model:",
            options=list(self.embedding_models.keys()),
            format_func=lambda x: self.embedding_models[x]["name"],
        )

        return llm, embedding


class SimplePDFProcessor:
    """Handle PDF processing and chunking"""

    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_pdf(self, pdf_file):
        """Read PDF and extract text"""
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
        return text

    def create_chunks(self, text, pdf_file):
        """Split text into chunks"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            if start > 0:
                start = start - self.chunk_overlap

            chunk = text[start:end]

            if end < len(text):
                last_period = chunk.rfind(".")
                if last_period != -1:
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1

            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {"source": getattr(pdf_file, "name", "uploaded.pdf")},
                }
            )

            start = end

        return chunks


class SimpleRAGSystem:
    """Simple RAG implementation with OpenAI, Ollama, and Gemini."""

    def __init__(self, embedding_model="openai", llm_model="openai"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # We do manual embeddings for Gemini (compute vectors ourselves)
        self.manual_embed = (self.embedding_model == "gemini")

        # Initialize ChromaDB
        self.db = chromadb.PersistentClient(path="./chroma_db")

        # Setup embedding function (only for non-Gemini)
        self.setup_embedding_function()

        # Setup LLM client(s)
        self.llm = None
        self.gemini_client = None
        self.chat_model = None
        self.setup_llm()

        # Get or create collection with proper handling
        self.collection = self.setup_collection()

    # ---------- Clients ----------

    def setup_llm(self):
        """Prepare LLM client(s) based on selection."""
        if self.llm_model == "openai":
            self.llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.chat_model = "gpt-4o-mini"

        elif self.llm_model == "ollama":
            # Use Ollama's OpenAI-compatible API
            self.llm = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
            self.chat_model = OLLAMA_MODEL
            # Warm-up to keep the model hot (best-effort)
            try:
                _ = self.llm.chat.completions.create(
                    model=self.chat_model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=4,
                    temperature=0,
                    timeout=30,
                    extra_body={"options": _ollama_options()},
                )
                st.sidebar.success("Ollama backend is responsive.")
            except Exception as e:
                st.sidebar.warning(f"Ollama warm-up: {e}")

        else:  # gemini
            if google_genai is None:
                st.error("Gemini selected but google-genai is not installed. Run: pip install google-genai")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY is not set in your .env")
            self.gemini_client = google_genai.Client(api_key=api_key)
            self.chat_model = "gemini-2.5-flash"

        # Ensure Gemini client exists if we need Gemini embeddings even when LLM != gemini
        if self.manual_embed and self.gemini_client is None:
            if google_genai is None:
                st.error("Gemini embeddings selected but google-genai is not installed. Run: pip install google-genai")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                st.error("GEMINI_API_KEY is not set in your .env")
            self.gemini_client = google_genai.Client(api_key=api_key)

    # ---------- Embeddings ----------

    def setup_embedding_function(self):
        """Setup the appropriate embedding function (non-Gemini)."""
        try:
            if self.embedding_model == "openai":
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small",
                )
            elif self.embedding_model == "nomic":
                # For Nomic embeddings via Ollama
                self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key="ollama",
                    api_base=OLLAMA_BASE_URL,
                    model_name="nomic-embed-text",
                )
            elif self.embedding_model == "chroma":
                self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            elif self.embedding_model == "gemini":
                # Manual: we compute vectors ourselves
                self.embedding_fn = None
            else:
                self.embedding_fn = None
        except Exception as e:
            st.error(f"Error setting up embedding function: {str(e)}")
            raise e

    def _embed_texts_gemini(self, texts):
        """Manual Gemini embeddings (returns list of vectors) with robust parsing."""
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized for embeddings.")

        def _first_float_list(obj):
            if isinstance(obj, dict):
                for k in ("values", "value", "embedding", "vector"):
                    v = obj.get(k)
                    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                        return [float(x) for x in v]
                    if isinstance(v, dict):
                        found = _first_float_list(v)
                        if found:
                            return found
                for v in obj.values():
                    found = _first_float_list(v)
                    if found:
                        return found
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    found = _first_float_list(v)
                    if found:
                        return found
            return None

        def _extract_vec(resp):
            emb = getattr(resp, "embedding", None)
            if emb is not None:
                vals = getattr(emb, "values", None) or getattr(emb, "value", None)
                if isinstance(vals, list) and vals and all(isinstance(x, (int, float)) for x in vals):
                    return [float(x) for x in vals]
            embs = getattr(resp, "embeddings", None)
            if isinstance(embs, (list, tuple)) and embs:
                first = embs[0]
                vals = getattr(first, "values", None) or getattr(first, "value", None)
                if isinstance(vals, list) and vals and all(isinstance(x, (int, float)) for x in vals):
                    return [float(x) for x in vals]
            if isinstance(resp, dict):
                found = _first_float_list(resp)
                if found:
                    return found
            to_dict = getattr(resp, "to_dict", None)
            if callable(to_dict):
                d = to_dict()
                found = _first_float_list(d)
                if found:
                    return found
            if hasattr(resp, "__dict__"):
                found = _first_float_list(resp.__dict__)
                if found:
                    return found
            return None

        vectors = []
        for t in texts:
            content_obj = {"parts": [{"text": t}]}
            try:
                resp = self.gemini_client.models.embed_content(
                    model="text-embedding-004", content=content_obj
                )
            except TypeError:
                try:
                    resp = self.gemini_client.models.embed_content(
                        model="text-embedding-004", contents=t
                    )
                except Exception:
                    resp = self.gemini_client.models.embed_content(
                        model="text-embedding-004", content=t
                    )

            vec = _extract_vec(resp)
            if not vec:
                raise RuntimeError("Unexpected Gemini embedding response format.")
            vectors.append(vec)
        return vectors

    # ---------- Vector store ----------

    def setup_collection(self):
        """Setup collection (omit embedding_function when using manual Gemini embeds)."""
        collection_name = f"documents_{self.embedding_model}"

        try:
            # Try to get existing collection first
            try:
                if self.embedding_fn is not None:
                    collection = self.db.get_collection(
                        name=collection_name, embedding_function=self.embedding_fn
                    )
                else:
                    collection = self.db.get_collection(name=collection_name)
                st.info(f"Using existing collection for {self.embedding_model} embeddings")
            except Exception:
                # If collection doesn't exist, create new one
                if self.embedding_fn is not None:
                    collection = self.db.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_fn,
                        metadata={"model": self.embedding_model},
                    )
                else:
                    collection = self.db.create_collection(
                        name=collection_name,
                        metadata={"model": self.embedding_model},
                    )
                st.success(f"Created new collection for {self.embedding_model} embeddings")

            return collection

        except Exception as e:
            st.error(f"Error setting up collection: {str(e)}")
            raise e

    def add_documents(self, chunks):
        """Add documents to ChromaDB (manual embeddings for Gemini)."""
        try:
            if not self.collection:
                self.collection = self.setup_collection()

            ids = [chunk["id"] for chunk in chunks]
            docs = [chunk["text"] for chunk in chunks]
            metas = [chunk["metadata"] for chunk in chunks]

            if self.manual_embed:
                embs = self._embed_texts_gemini(docs)
                self.collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            else:
                self.collection.add(ids=ids, documents=docs, metadatas=metas)

            return True
        except Exception as e:
            st.error(f"Error adding documents: {str(e)}")
            return False

    def query_documents(self, query, n_results=3):
        """Query documents and return relevant chunks (manual embeds for Gemini)."""
        try:
            if not self.collection:
                raise ValueError("No collection available")

            if self.manual_embed:
                qvec = self._embed_texts_gemini([query])[0]
                results = self.collection.query(query_embeddings=[qvec], n_results=n_results)
            else:
                results = self.collection.query(query_texts=[query], n_results=n_results)

            return results
        except Exception as e:
            st.error(f"Error querying documents: {str(e)}")
            return None

    # ---------- Chat ----------

    def _ollama_chat(self, messages):
        """Call Ollama with long timeout, retries, and keep_alive options."""
        last_err = None
        for attempt in range(OLLAMA_RETRIES):
            try:
                resp = self.llm.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512,
                    timeout=OLLAMA_TIMEOUT_S,
                    extra_body={"options": _ollama_options()},
                )
                return resp
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "timeout" in msg and attempt < OLLAMA_RETRIES - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s backoff
                    continue
                raise last_err

    def generate_response(self, query, context):
        """Generate response using selected LLM"""
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            If you can't find the answer in the context, say so, or I don't know.

            Context: {context}

            Question: {query}

            Answer:
            """.strip()

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]

            if self.llm_model == "openai":
                response = self.llm.chat.completions.create(
                    model=self.chat_model,
                    messages=messages,
                )
                return response.choices[0].message.content

            elif self.llm_model == "ollama":
                response = self._ollama_chat(messages)
                return response.choices[0].message.content

            else:  # gemini
                resp = self.gemini_client.models.generate_content(
                    model=self.chat_model,
                    contents=prompt
                )
                return getattr(resp, "text", None) or "I don't know"

        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None

    def get_embedding_info(self):
        """Get information about current embedding model"""
        model_selector = SimpleModelSelector()
        model_info = model_selector.embedding_models[self.embedding_model]
        return {
            "name": model_info["name"],
            "dimensions": model_info["dimensions"],
            "model": self.embedding_model,
        }


def main():
    st.title("🤖 Simple RAG System")

    # Initialize session state
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_embedding_model" not in st.session_state:
        st.session_state.current_embedding_model = None
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None

    # Initialize model selector
    model_selector = SimpleModelSelector()
    llm_model, embedding_model = model_selector.select_models()

    # Check if embedding model changed
    if embedding_model != st.session_state.current_embedding_model:
        st.session_state.processed_files.clear()  # Clear processed files
        st.session_state.current_embedding_model = embedding_model
        st.session_state.rag_system = None  # Reset RAG system
        st.warning("Embedding model changed. Please re-upload your documents.")

    # Initialize RAG system
    try:
        if st.session_state.rag_system is None:
            st.session_state.rag_system = SimpleRAGSystem(embedding_model, llm_model)

        # Display current embedding model info
        embedding_info = st.session_state.rag_system.get_embedding_info()
        st.sidebar.info(
            f"Current Embedding Model:\n"
            f"- Name: {embedding_info['name']}\n"
            f"- Dimensions: {embedding_info['dimensions']}"
        )
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return

    # File upload
    pdf_file = st.file_uploader("Upload PDF", type="pdf")

    if pdf_file and pdf_file.name not in st.session_state.processed_files:
        # Process PDF
        processor = SimplePDFProcessor()
        with st.spinner("Processing PDF..."):
            try:
                # Extract text
                text = processor.read_pdf(pdf_file)
                # Create chunks
                chunks = processor.create_chunks(text, pdf_file)
                # Add to database
                if st.session_state.rag_system.add_documents(chunks):
                    st.session_state.processed_files.add(pdf_file.name)
                    st.success(f"Successfully processed {pdf_file.name}")
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # Query interface
    if st.session_state.processed_files:
        st.markdown("---")
        st.subheader("🔍 Query Your Documents")
        query = st.text_input("Ask a question:")

        if query:
            with st.spinner("Generating response..."):
                # Get relevant chunks
                results = st.session_state.rag_system.query_documents(query)
                if results and results.get("documents"):
                    # Generate response
                    response = st.session_state.rag_system.generate_response(
                        query, results["documents"][0]
                    )

                    if response:
                        # Display results
                        st.markdown("### 📝 Answer:")
                        st.write(response)

                        with st.expander("View Source Passages"):
                            for idx, doc in enumerate(results["documents"][0], 1):
                                st.markdown(f"**Passage {idx}:**")
                                st.info(doc)
                else:
                    st.warning("No relevant passages found. Try rephrasing your question.")
    else:
        st.info("👆 Please upload a PDF document to get started!")


if __name__ == "__main__":
    main()
