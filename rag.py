"""논문 다운로드, PDF 파싱·청킹, 임베딩, Vector DB, 검색. (담당: 기술 조사)"""
from __future__ import annotations

import hashlib
import re
import urllib.request

import chromadb
import pymupdf as fitz  # PyMuPDF: fitz는 구버전 별칭이라 경고가 발생해 새 alias로 import

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, PAPERS, TOP_K, openai_client

_collection = None


def download_papers() -> dict[str, int]:
    page_counts = {}

    for technology, info in PAPERS.items():
        path = info["path"]

        if not path.exists():
            print(f"[DOWNLOAD] {technology}: {info['url']}")
            urllib.request.urlretrieve(info["url"], path)

        with fitz.open(path) as document:
            page_counts[technology] = len(document)

    total_pages = sum(page_counts.values())
    print("문서별 페이지:", page_counts)
    print("전체 페이지:", total_pages)

    if total_pages > 200:
        raise ValueError(f"문서 풀이 200페이지를 초과했습니다: {total_pages}")

    return page_counts


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, max_chars: int = 1400, overlap: int = 150) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def load_and_chunk_papers() -> list[dict]:
    chunks = []

    for technology, info in PAPERS.items():
        with fitz.open(info["path"]) as document:
            for page_number, page in enumerate(document, start=1):
                text = normalize_text(page.get_text("text"))
                for chunk_number, chunk_text in enumerate(split_text(text), start=1):
                    raw_id = f"{technology}|{page_number}|{chunk_number}|{chunk_text[:80]}"
                    chunk_id = hashlib.sha256(raw_id.encode()).hexdigest()[:20]
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "technology": technology,
                        "page": page_number,
                        "file_name": info["path"].name,
                        "source_url": info["url"],
                    })

    return chunks


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def build_index(chunks: list[dict]):
    global _collection
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )

    batch_size = 16
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        embeddings = embed_texts([item["text"] for item in batch])
        collection.upsert(
            ids=[item["chunk_id"] for item in batch],
            documents=[item["text"] for item in batch],
            embeddings=embeddings,
            metadatas=[{
                "technology": item["technology"],
                "page": item["page"],
                "file_name": item["file_name"],
                "source_url": item["source_url"],
            } for item in batch],
        )

    _collection = collection
    return collection


def get_collection():
    """build_index() 실행 후 생성된 컬렉션을 반환합니다."""
    if _collection is None:
        raise RuntimeError("build_index()를 먼저 실행하세요.")
    return _collection


def retrieve(query: str, technology: str | None = None, top_k: int = TOP_K) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    ).data[0].embedding

    where = {"technology": technology} if technology else None
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    items = []
    for chunk_id, text, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        items.append({
            "chunk_id": chunk_id,
            "text": text,
            "similarity": 1 - float(distance),
            **metadata,
        })

    return items
