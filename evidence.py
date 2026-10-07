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
            source_tier=1,
            source_category="primary_research",
            publisher="arXiv",
            is_independent=True,
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


def _format_citation(item: dict, pages: list[int] | None = None) -> str:
    """노션 가이드의 REFERENCE 표기 형식에 맞춰 논문/웹 항목을 각각 포맷합니다.
    논문: 저자(YYYY). 논문제목. arXiv:ID. (p.페이지)
    웹  : 사이트명(YYYY-MM-DD). 제목, URL (발행일이 없으면 날짜 미상)
    """
    if item.get("source_type") == "paper":
        paper = PAPERS.get(item.get("technology"), {})
        authors, year, arxiv_id, title = (
            paper.get("authors"), paper.get("year"), paper.get("arxiv_id"), paper.get("title"),
        )
        page_numbers = sorted(set(pages or ([item["page"]] if item.get("page") else [])))
        if len(page_numbers) == 1:
            page = f" (p.{page_numbers[0]})"
        elif page_numbers:
            page = f" (pp.{', '.join(str(value) for value in page_numbers)})"
        else:
            page = ""
        if authors and year and arxiv_id and title:
            return f"{authors}({year}). {title}. arXiv:{arxiv_id}.{page}"
        # 서지정보를 확인하지 못한 논문은 임의로 지어내지 않고 최소 정보만 표기
        location = item.get("url") or item.get("file_name", "")
        return f"{item['title']}{page}. {location}"

    # 웹 자료: 사이트명은 URL 도메인에서 도출(작성자/기관명은 원자료에서 직접 확인 필요)
    url = item.get("url") or ""
    site_name = urlparse(url).netloc or "출처 미상"
    published_date = item.get("published_at")
    date_label = f"({published_date or '날짜 미상'})"
    return f"{site_name}{date_label}. {item['title']}, {url}"


def format_references(references: list[dict], used_ids: set[str] | None = None) -> str:
    grouped: dict[tuple, dict] = {}
    pages_by_key: dict[tuple, list[int]] = {}
    for item in references:
        evidence_id = item.get("evidence_id")
        if used_ids is not None and evidence_id not in used_ids:
            continue
        if item.get("source_type") == "paper":
            key = ("paper", item.get("url") or item.get("file_name"))
        elif item.get("source_type") == "web":
            key = ("web", item.get("url"))
        else:
            key = evidence_id or f"{item.get('url')}:{item.get('page')}"
        grouped.setdefault(key, item)
        if item.get("page") is not None:
            pages_by_key.setdefault(key, []).append(int(item["page"]))
    lines = [
        f"- [{item['evidence_id']}] {_format_citation(item, pages_by_key.get(key))}"
        for key, item in grouped.items()
    ]
    return "\n".join(lines) or "- 실제 활용 자료 없음"


def _reference_key(item: dict) -> tuple:
    if item.get("source_type") == "paper":
        return ("paper", item.get("url") or item.get("file_name") or item.get("title"))
    if item.get("source_type") == "web":
        return ("web", item.get("url") or item.get("title"))
    return (item.get("source_type"), item.get("url") or item.get("file_name") or item.get("title"))


def build_numbered_references(
    references: list[dict],
    used_ids: set[str],
) -> tuple[dict[str, int], str, dict[str, list[str]]]:
    """사용된 Evidence를 출처 단위로 묶어 최종 숫자 인용을 만든다."""
    groups: dict[tuple, list[dict]] = {}
    for item in references:
        evidence_id = item.get("evidence_id")
        if evidence_id not in used_ids:
            continue
        groups.setdefault(_reference_key(item), []).append(item)

    evidence_to_number: dict[str, int] = {}
    citation_map: dict[str, list[str]] = {}
    lines = []
    for number, items in enumerate(groups.values(), start=1):
        evidence_ids = list(dict.fromkeys(
            item["evidence_id"] for item in items if item.get("evidence_id")
        ))
        for evidence_id in evidence_ids:
            evidence_to_number[evidence_id] = number
        citation_map[str(number)] = evidence_ids
        pages = [int(item["page"]) for item in items if item.get("page") is not None]
        lines.append(f"- [{number}] {_format_citation(items[0], pages)}")

    return evidence_to_number, "\n".join(lines) or "- 실제 활용 자료 없음", citation_map
