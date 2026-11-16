from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.config.config import DB_FAISS_PATH
import os

logger = get_logger(__name__)

def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading vector store from %s", DB_FAISS_PATH)
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.warning("No vector store found at %s", DB_FAISS_PATH)
            return None
    except Exception:
        logger.exception("Failed to load vector store from %s", DB_FAISS_PATH)
        raise


def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise ValueError("No text chunks were provided for indexing")
        logger.info("Building vector store at %s", DB_FAISS_PATH)
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        logger.info("Vector stored successfully")
        return db
    except Exception:
        logger.exception("Failed to create vector store at %s", DB_FAISS_PATH)
        raise
