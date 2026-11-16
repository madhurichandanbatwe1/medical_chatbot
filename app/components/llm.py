from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY, GROQ_MODEL
from app.common.logger import get_logger

logger = get_logger(__name__)


def load_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY must be configured")

    logger.info("Loading LLM from Groq")
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=1024,
    )
    logger.info("Groq LLM configured successfully")
    return llm