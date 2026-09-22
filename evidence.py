"""Evidence 생성·요약·참고문헌 포맷 공통 유틸."""
from __future__ import annotations

import hashlib
from urllib.parse import urlparse

from config import FAST_MODE, PAPERS, TOP_K
from rag import retrieve
from state import Evidence


def make_evidence_id(prefix: str, value: str) -> str:
    return f"{prefix}-{hashlib.sha256(value.encode()).hexdigest()[:12]}"


def rag_evidence(query: str, technology: str, agent: str, top_k: int = TOP_K) -> list[dict]:
    evidence = []
    for item in retrieve(query, technology=technology, top_k=top_k):
        # 동일 청크가 여러 질의에 검색되어도 하나의 논문 근거로 취급합니다.
        evidence_id = make_evidence_id("rag", f"{technology}|{item['chunk_id']}")
        evidence.append(Evidence(
            evidence_id=evidence_id,
            agent=agent,
            technology=technology,
            source_type="paper",
            title=item["file_name"],
            claim=query,
            evidence_text=item["text"],
            url=item.get("source_url"),
            file_name=item["file_name"],
            page=int(item["page"]),
            chunk_id=item["chunk_id"],
            retrieval_score=item["similarity"],
        ).model_dump())
    return evidence


def compact_evidence(evidence: list[dict], max_chars: int | None = None) -> str:
    max_chars = max_chars or (10000 if FAST_MODE else 24000)
    unique = {}
    for item in evidence:
        unique[item.get("evidence_id") or repr(item)] = item
    groups: dict[str, list[dict]] = {}
    for item in unique.values():
        groups.setdefault(item.get("technology", "unknown"), []).append(item)

    blocks = []
    length = 0
    budget = max_chars // max(len(groups), 1)
    for group_items in groups.values():
        group_length = 0
        for item in group_items:
            block = (
                f"[{item['evidence_id']}] {item['title']}"
                f" | page={item.get('page')} | url={item.get('url')}\n"
                f"{item['evidence_text'][:3000]}"
            )
            if group_length + len(block) > budget:
                continue
            blocks.append(block)
            group_length += len(block)
            length += len(block)

    for item in unique.values():
        block = (
            f"[{item['evidence_id']}] {item['title']}"
            f" | page={item.get('page')} | url={item.get('url')}\n"
            f"{item['evidence_text'][:3000]}"
        )
        if block not in blocks and length + len(block) <= max_chars:
            blocks.append(block)
            length += len(block)
    return "\n\n".join(blocks)


def collect_evidence_ids(value) -> set[str]:
    ids = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"evidence_ids", "evidence_id"}:
                if isinstance(item, list):
                    ids.update(str(x) for x in item)
                elif isinstance(item, str):
                    ids.add(item)
            else:
                ids.update(collect_evidence_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.update(collect_evidence_ids(item))
    return ids


def _format_citation(item: dict) -> str:
    """노션 가이드의 REFERENCE 표기 형식에 맞춰 논문/웹 항목을 각각 포맷합니다.
    논문: 저자(YYYY). 논문제목. arXiv:ID. (p.페이지)
    웹  : 사이트명(YYYY-MM-DD). 제목, URL (발행일이 없으면 날짜 생략)
    """
    if item.get("source_type") == "paper":
        paper = PAPERS.get(item.get("technology"), {})
        authors, year, arxiv_id, title = (
            paper.get("authors"), paper.get("year"), paper.get("arxiv_id"), paper.get("title"),
        )
        page = f" (p.{item['page']})" if item.get("page") else ""
        if authors and year and arxiv_id and title:
            return f"{authors}({year}). {title}. arXiv:{arxiv_id}.{page}"
        # 서지정보를 확인하지 못한 논문은 임의로 지어내지 않고 최소 정보만 표기
        location = item.get("url") or item.get("file_name", "")
        return f"{item['title']}{page}. {location}"

    # 웹 자료: 사이트명은 URL 도메인에서 도출(작성자/기관명은 원자료에서 직접 확인 필요)
    url = item.get("url") or ""
    site_name = urlparse(url).netloc or "출처 미상"
    published_date = item.get("published_at")
    date_label = f"({published_date})" if published_date else ""
    return f"{site_name}{date_label}. {item['title']}, {url}"


def format_references(references: list[dict], used_ids: set[str] | None = None) -> str:
    lines = []
    seen = set()
    for item in references:
        evidence_id = item.get("evidence_id")
        if used_ids is not None and evidence_id not in used_ids:
            continue
        key = evidence_id or f"{item.get('url')}:{item.get('page')}"
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- [{item['evidence_id']}] {_format_citation(item)}")
    return "\n".join(lines) or "- 실제 활용 자료 없음"
