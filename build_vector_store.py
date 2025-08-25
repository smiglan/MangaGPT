"""Utility to scrape chapter pages and build a FAISS vector store."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Set

import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


CONFIG_PATH = Path("manga_config.json")
INDEX_ROOT = Path("vector_store")

with CONFIG_PATH.open() as fh:
    MANGA_CONFIG = json.load(fh)


def request_with_retry(
    method: str, url: str, retries: int = 3, backoff: float = 1.0, **kwargs
) -> requests.Response:
    """Perform an HTTP request with simple exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.request(method, url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))


def fetch_text(url: str) -> str:
    """Download a web page and return plain text."""
    response = request_with_retry("GET", url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text("\n", strip=True)


def build_vector_store(urls: Iterable[str], index_path: Path) -> None:
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
        try:
            text = fetch_text(url)
        except requests.RequestException as exc:
            print(f"Skipping {url}: {exc}")
            continue
        new_texts.append(text)
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


def fetch_chapter_links(manga: str, start: int = 1) -> List[str]:
    """Return links for all chapters of ``manga`` starting from ``start``."""
    base = MANGA_CONFIG[manga]
    links: List[str] = []
    num = start
    while True:
        url = base.format(num)
        try:
            request_with_retry("HEAD", url)
        except requests.RequestException:
            break
        links.append(url)
        num += 1
    return links


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manga",
        choices=MANGA_CONFIG.keys(),
        default="one_piece",
        help="Key of the manga to process",
    )
    args = parser.parse_args()

    index_path = INDEX_ROOT / args.manga

    # Determine the next chapter to fetch based on the existing index
    start_chapter = 1
    if index_path.exists():
        embeddings = OpenAIEmbeddings()
        store = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        sources = [doc.metadata.get("source") for doc in store.docstore._dict.values()]
        chapters: List[int] = []
        for s in sources:
            if s:
                chapter = s.rsplit("_", 1)[-1]
                if chapter.isdigit():
                    chapters.append(int(chapter))
        start_chapter = max(chapters, default=0) + 1

    links = fetch_chapter_links(args.manga, start_chapter)
    build_vector_store(links, index_path)
    print(f"Vector store written to {index_path}/")
