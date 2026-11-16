import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.components.pdfloader import create_text_chunks


class AlternatingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if index % 2 == 0 else [0.0, 1.0]
            for index, _ in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class SemanticChunkingTests(unittest.TestCase):
    def test_semantic_chunks_preserve_source_metadata(self):
        source = Document(
            page_content=(
                "First medical sentence. Second medical sentence. "
                "Third medical sentence. Fourth medical sentence."
            ),
            metadata={"source": "medical.pdf", "page": 12},
        )

        with patch(
            "app.components.pdfloader.get_embedding_model",
            return_value=AlternatingEmbeddings(),
        ):
            chunks = create_text_chunks([source])

        self.assertTrue(chunks)
        self.assertTrue(
            all(
                chunk.metadata["source"] == "medical.pdf"
                and chunk.metadata["page"] == 12
                for chunk in chunks
            )
        )


if __name__ == "__main__":
    unittest.main()
