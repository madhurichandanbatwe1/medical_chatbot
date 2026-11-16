import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.config.config import DATA_PATH

logger = get_logger(__name__)

def load_pdf_file():
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data path does not exist: {DATA_PATH}")
        logger.info(f"Loading files from {DATA_PATH}")
        loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        if not documents:
            raise FileNotFoundError(f"No PDF documents found in {DATA_PATH}")

        logger.info("Successfully fetched %s documents", len(documents))
        return documents

    except Exception:
        logger.exception("Failed to load PDF files")
        raise


def create_text_chunks(documents):
    try:
        if not documents:
            raise ValueError("No documents were provided for chunking")
        logger.info("Semantically splitting %s documents into chunks", len(documents))
        text_splitter = SemanticChunker(
            embeddings=get_embedding_model(),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=100,
        )
        text_chunks = text_splitter.split_documents(documents)
        if not text_chunks:
            raise ValueError("No text chunks were generated from the documents")
        logger.info("Generated %s text chunks", len(text_chunks))
        return text_chunks

    except Exception:
        logger.exception("Failed to generate text chunks")
        raise