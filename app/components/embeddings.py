from langchain_ollama import OllamaEmbeddings
from app.common.logger import get_logger
from app.config.config import OLLAMA_BASE_URL, OLLAMA_EMBEDDING_MODEL


logger = get_logger(__name__)


class NomicOllamaEmbeddings(OllamaEmbeddings):
    batch_size: int = 32

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed_texts = [f"search_document: {text}" for text in texts]
        embeddings = []
        for start in range(0, len(prefixed_texts), self.batch_size):
            batch = prefixed_texts[start : start + self.batch_size]
            embeddings.extend(super().embed_documents(batch))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"search_query: {text}")


def get_embedding_model() -> NomicOllamaEmbeddings:
    logger.info("Initializing Ollama embedding model %s", OLLAMA_EMBEDDING_MODEL)
    model = NomicOllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    logger.info("Ollama embedding model configured")
    return model
    