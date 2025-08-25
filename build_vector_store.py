"""Utility to scrape chapter pages and build a FAISS vector store."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set

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
    ``index_path``. When ``index_path`` already exists, only new chapters
    (based on their source URL) are embedded and appended to the store.
    """

    embeddings = OpenAIEmbeddings()

    # Load existing index if present to avoid re-processing chapters
    if index_path.exists():
        store = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
        existing: Set[str] = {doc.metadata.get("source") for doc in store.docstore._dict.values()}
    else:
        store = None
        existing = set()

    # Fetch and prepare documents for URLs not yet indexed
    new_texts: List[str] = []
    new_meta: List[dict] = []
    for url in urls:
        if url in existing:
            continue
        new_texts.append(fetch_text(url))
        new_meta.append({"source": url})

    if not new_texts:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents(new_texts, metadatas=new_meta)

    if store is None:
        store = FAISS.from_documents(docs, embeddings)
    else:
        store.add_documents(docs)

    store.save_local(str(index_path))


def fetch_chapter_links(start: int = 1) -> List[str]:
    """Return links for all chapters starting from ``start``.

    The function sequentially probes chapter pages until a 404 response is
    encountered, yielding the set of available chapter URLs.
    """
    base = "https://onepiece.fandom.com/wiki/Chapter_{}"
    links: List[str] = []
    num = start
    while True:
        url = base.format(num)
        resp = requests.head(url, timeout=30)
        if resp.status_code == 200:
            links.append(url)
            num += 1
        else:
            break
    return links


if __name__ == "__main__":
    # Determine the next chapter to fetch based on the existing index
    start_chapter = 1
    if DEFAULT_INDEX_PATH.exists():
        embeddings = OpenAIEmbeddings()
        store = FAISS.load_local(
            str(DEFAULT_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        sources = [doc.metadata.get("source") for doc in store.docstore._dict.values()]
        chapters = [int(s.rsplit("_", 1)[-1]) for s in sources if s and s.rsplit("_", 1)[-1].isdigit()]
        start_chapter = max(chapters, default=0) + 1

    links = fetch_chapter_links(start_chapter)
    build_vector_store(links)
    print(f"Vector store written to {DEFAULT_INDEX_PATH}/")
