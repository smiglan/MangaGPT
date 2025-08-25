"""Streamlit interface for asking questions about scraped chapters."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

INDEX_PATH = Path("vector_store")

@st.cache_resource
def load_store() -> FAISS:
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(str(INDEX_PATH), embeddings)


def main() -> None:
    st.title("MangaGPT Q&A")
    question = st.text_input("Ask a question about the manga")
    if st.button("Answer") and question:
        store = load_store()
        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(model_name="gpt-5-nano"),
            retriever=store.as_retriever(),
        )
        answer = qa_chain.run(question)
        st.write(answer)


if __name__ == "__main__":
    main()
