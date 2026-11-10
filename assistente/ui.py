import hashlib

import streamlit as st

from assistente.assistant import answer_question, create_chat_model
from assistente.config import get_settings
from assistente.rag import create_embeddings, create_retriever, index_pdf


@st.cache_resource(show_spinner="Preparando o modelo de busca…")
def cached_embeddings():
    return create_embeddings()


@st.cache_resource(show_spinner=False)
def cached_chat_model(api_key: str, model: str):
    return create_chat_model(api_key, model)


def _render_chat() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def run_app() -> None:
    st.set_page_config(page_title="Assistente de Escritório", page_icon="⚖️", layout="wide")
    settings = get_settings()

    st.title("⚖️ Assistente de Escritório")
    st.caption("Converse, organize ideias e consulte documentos PDF quando precisar.")

    with st.sidebar:
        st.header("Documento de referência")
        pdf_file = st.file_uploader("Envie um PDF", type=["pdf"])
        st.caption("O conteúdo enviado é processado nesta sessão para responder às suas perguntas.")
        st.divider()
        st.info("Respostas geradas por IA podem conter erros. Confira informações importantes.")

    if not settings.groq_api_key:
        st.error("Configure GROQ_API_KEY no arquivo .env para iniciar o assistente.")
        st.code("GROQ_API_KEY=sua_chave\nGROQ_MODEL=openai/gpt-oss-20b", language="text")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_fingerprint" not in st.session_state:
        st.session_state.pdf_fingerprint = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if pdf_file:
        pdf_bytes = pdf_file.getvalue()
        fingerprint = hashlib.sha256(pdf_bytes).hexdigest()
        if fingerprint != st.session_state.pdf_fingerprint:
            try:
                with st.spinner("Lendo e indexando o PDF…"):
                    indexed_fingerprint, vector_store = index_pdf(
                        pdf_bytes, pdf_file.name, cached_embeddings()
                    )
                st.session_state.pdf_fingerprint = indexed_fingerprint
                st.session_state.vector_store = vector_store
                st.success(f"Documento pronto para consulta: {pdf_file.name}")
            except Exception as error:
                st.session_state.pdf_fingerprint = None
                st.session_state.vector_store = None
                st.error(f"Não foi possível processar o PDF: {error}")
    elif st.session_state.pdf_fingerprint:
        st.session_state.pdf_fingerprint = None
        st.session_state.vector_store = None

    _render_chat()
    question = st.chat_input("Como posso ajudar?")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Preparando uma resposta…"):
                    answer = answer_question(
                        cached_chat_model(settings.groq_api_key, settings.groq_model),
                        question,
                        st.session_state.messages[:-1],
                        create_retriever(st.session_state.vector_store)
                        if st.session_state.vector_store is not None
                        else None,
                    )
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as error:
                st.error(f"Não consegui responder agora: {error}")
        if len(st.session_state.messages) > 24:
            st.session_state.messages = st.session_state.messages[-24:]
