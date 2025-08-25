"""Streamlit interface for asking questions about scraped chapters."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

INDEX_ROOT = Path("vector_store")


@st.cache_resource
def load_store(manga: str) -> FAISS:
    """Load the local FAISS vector store for ``manga``."""
    index_path = INDEX_ROOT / manga
    if not index_path.exists():
        raise FileNotFoundError(
            f"Vector store for {manga} not found. Run `build_vector_store.py --manga {manga}` first."
        )
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )


def main() -> None:
    st.title("MangaGPT Q&A")

    if not INDEX_ROOT.exists():
        st.error("No vector stores found. Run `build_vector_store.py` first.")
        return

    mangas = [p.name for p in INDEX_ROOT.iterdir() if p.is_dir()]
    if not mangas:
        st.error("No vector stores found. Run `build_vector_store.py` first.")
        return

    manga = st.selectbox("Choose manga", mangas)
    question = st.text_input("Ask a question about the manga")
    if st.button("Answer") and question:
        store = load_store(manga)
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(model_name="gpt-4o-mini"),
            retriever=store.as_retriever(),
        )
        answer = qa_chain.run(question)
        st.write(answer)


if __name__ == "__main__":
    main()
