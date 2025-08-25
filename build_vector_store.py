"""Utility to scrape chapter pages and build a FAISS vector store."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


DEFAULT_INDEX_PATH = Path("vector_store")


def fetch_text(url: str) -> str:
    """Download a web page and return plain text.

    Parameters
    ----------
    url: str
        Web page URL.
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text("\n", strip=True)


def build_vector_store(urls: Iterable[str], index_path: Path = DEFAULT_INDEX_PATH) -> None:
    """Fetch all URLs and persist a FAISS vector store locally.

    The function splits the downloaded text into overlapping chunks and
    computes OpenAI embeddings for each chunk before saving the index to
    ``index_path``.
    """
    texts = [fetch_text(u) for u in urls]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(texts)
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs, embeddings)
    store.save_local(str(index_path))


if __name__ == "__main__":
    # Example usage for manual runs
    links = [
        "https://onepiece.fandom.com/wiki/Chapter_1140",
    ]
    build_vector_store(links)
    print(f"Vector store written to {DEFAULT_INDEX_PATH}/")
