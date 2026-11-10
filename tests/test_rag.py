import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from assistente.rag import create_retriever, create_vector_store, format_documents, split_documents


class KeywordEmbeddings(Embeddings):
    vocabulary = ("rescisão", "multa", "férias")

    def _embed(self, text: str) -> list[float]:
        normalized = text.casefold()
        return [float(term in normalized) for term in self.vocabulary]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class RagTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            Document(
                page_content="A cláusula prevê rescisão mediante aviso prévio.",
                metadata={"source": "contrato.pdf", "page": 0},
            ),
            Document(
                page_content="A multa contratual será calculada sobre o saldo devido.",
                metadata={"source": "contrato.pdf", "page": 1},
            ),
            Document(
                page_content="O período de férias será definido entre as partes.",
                metadata={"source": "contrato.pdf", "page": 2},
            ),
        ]

    def test_split_preserves_page_and_adds_one_based_page_number(self):
        chunks = split_documents(self.documents, chunk_size=40, chunk_overlap=5)

        self.assertTrue(chunks)
        self.assertTrue(all("page_number" in chunk.metadata for chunk in chunks))
        self.assertEqual(chunks[0].metadata["page_number"], 1)

    def test_vector_retriever_returns_relevant_document(self):
        vector_store = create_vector_store(self.documents, KeywordEmbeddings())
        retriever = create_retriever(vector_store, k=2)

        results = retriever.invoke("rescisão")

        self.assertTrue(results)
        self.assertIn("rescisão", results[0].page_content)

    def test_format_includes_source_and_human_page_number(self):
        formatted = format_documents(self.documents[:1])

        self.assertIn("contrato.pdf, p. 1", formatted)
        self.assertIn("rescisão", formatted)


if __name__ == "__main__":
    unittest.main()
