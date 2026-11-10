import hashlib
import tempfile
from pathlib import Path
from typing import Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(
    documents: Sequence[Document], chunk_size: int = 900, chunk_overlap: int = 140
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))
    for chunk in chunks:
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = int(chunk.metadata["page"]) + 1
    return chunks


def load_pdf(pdf_bytes: bytes, filename: str = "documento.pdf") -> list[Document]:
    if not pdf_bytes:
        raise ValueError("O PDF está vazio.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
            temporary_file.write(pdf_bytes)
            temporary_path = Path(temporary_file.name)

        documents = PyPDFLoader(str(temporary_path)).load()
        for document in documents:
            document.metadata["source"] = filename
        if not any(document.page_content.strip() for document in documents):
            raise ValueError("Não encontrei texto extraível. O PDF pode ser uma digitalização.")
        return documents
    finally:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)


def create_vector_store(
    documents: Sequence[Document], embeddings: Embeddings
) -> Chroma:
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("Não foi possível extrair trechos pesquisáveis do PDF.")
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def index_pdf(pdf_bytes: bytes, filename: str, embeddings: Embeddings) -> tuple[str, Chroma]:
    fingerprint = hashlib.sha256(pdf_bytes).hexdigest()
    documents = load_pdf(pdf_bytes, filename)
    return fingerprint, create_vector_store(documents, embeddings)


def create_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(12, k * 3), "lambda_mult": 0.65},
    )


def format_documents(documents: Sequence[Document]) -> str:
    excerpts = []
    for document in documents:
        metadata = document.metadata or {}
        page = metadata.get("page_number")
        if page is None:
            source_page = metadata.get("page")
            page = int(source_page) + 1 if source_page is not None else "?"
        source = metadata.get("source", "PDF enviado")
        content = " ".join(document.page_content.split())
        if len(content) > 1100:
            content = content[:1100].rsplit(" ", 1)[0] + "…"
        excerpts.append(f"[{source}, p. {page}] {content}")
    return "\n\n".join(excerpts)
