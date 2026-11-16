from statistics import median
from typing import Any

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from sentence_transformers import CrossEncoder
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger

logger = get_logger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_CANDIDATE_LIMIT = 10

CUSTOM_PROMPT_TEMPLATE = '''Answer the medical question using relevant evidence from the context.
Address all requested aspects with a clear, sufficiently detailed explanation.
Do not add unsupported medical claims or present associations as causes.
If the context is missing or unclear for part of the question, identify that limitation instead of guessing.
Context:
{context}
Question:
{input}
Answer: 

'''

def set_custom_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "input"],
    )


def retrieve_and_rerank(
    question: str,
    vector_store: FAISS,
    reranker: CrossEncoder,
) -> list[Document]:
    scored_documents = vector_store.similarity_search_with_score(
        question,
        k=RETRIEVAL_CANDIDATE_LIMIT,
    )
    if not scored_documents:
        return []

    distances = [float(score) for _, score in scored_documents]
    median_distance = median(distances)
    median_deviation = median(abs(distance - median_distance) for distance in distances)
    distance_cutoff = median_distance + median_deviation
    documents = [
        document
        for document, distance in scored_documents
        if float(distance) <= distance_cutoff
    ]

    pairs = [(question, document.page_content) for document in documents]
    scores = reranker.predict(pairs)
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )
    return [documents[index] for index in ranked_indices]


def create_qa_chain():
    logger.info("Loading vector store for context")
    db = load_vector_store()
    if db is None:
        raise RuntimeError("Vector store is not present or is empty")

    llm = load_llm()
    reranker = CrossEncoder(RERANKER_MODEL)

    def rerank_input(inputs: dict[str, Any]) -> list[Document]:
        return retrieve_and_rerank(
            inputs["input"],
            db,
            reranker,
        )

    reranked_retriever = RunnableLambda(rerank_input)
    prompt = set_custom_prompt_template()
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(reranked_retriever, combine_docs_chain)
    logger.info("RAG chain created successfully")
    return rag_chain