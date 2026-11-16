from app.components.pdfloader import load_pdf_file,create_text_chunks
from app.components.vector_store import save_vector_store
from app.common.logger import get_logger

logger=get_logger(__name__)

def process_and_store_pdf():
    try:
        logger.info("Building the vector store")
        documents = load_pdf_file()
        text_chunks = create_text_chunks(documents)
        vector_store = save_vector_store(text_chunks)
        logger.info("Vector Store created successfully")
        return vector_store
    except Exception:
        logger.exception("Failed to create vector store")
        raise


if __name__=="__main__":
    process_and_store_pdf()
            

