import unittest

from langchain_core.documents import Document

from app.components.retriever import retrieve_and_rerank


class RetrieveAndRerankTests(unittest.TestCase):
    def test_dynamically_keeps_relevant_chunks_and_reranks_them(self):
        documents = [
            Document(page_content="Symptoms information"),
            Document(page_content="Heredity information"),
            Document(page_content="Related symptoms information"),
            Document(page_content="Unrelated information"),
            Document(page_content="Distant information"),
        ]

        class StubVectorStore:
            def similarity_search_with_score(self, question, k):
                self.requested_k = k
                return list(zip(documents, [0.29, 0.39, 0.34, 0.6, 0.8]))

        class StubReranker:
            def predict(self, pairs):
                return [0.8, 0.2, 0.5]

        vector_store = StubVectorStore()
        result = retrieve_and_rerank(
            "What are symptoms and does heredity matter?",
            vector_store,
            StubReranker(),
        )

        self.assertEqual(vector_store.requested_k, 10)
        self.assertEqual(
            [document.page_content for document in result],
            [
                "Symptoms information",
                "Related symptoms information",
                "Heredity information",
            ],
        )


if __name__ == "__main__":
    unittest.main()
