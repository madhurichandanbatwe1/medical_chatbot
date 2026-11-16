import os

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_FAISS_PATH = "vectorstore/db_faiss_nomic_semantic"
DATA_PATH = "data/"