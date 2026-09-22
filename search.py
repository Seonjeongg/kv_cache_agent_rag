"""외부 웹 검색: Tavily -> DDGS -> DuckDuckGo HTML 순서의 대체 경로."""
from __future__ import annotations

import json
import re
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from config import FETCH_WEB_FULL_TEXT, tavily_client
from evidence import make_evidence_id
from rag import normalize_text
from state import Evidence


def is_relevant_result(title: str, body: str, url: str, technology: str) -> bool:
    text = f"{title} {body} {url}".lower()
    keyword_groups = {
        "DeepSeek-V2 MLA": ("deepseek", "mla", "multi-head latent", "key-value cache"),
        "ITME": ("itme", "cxl", "hybrid memory", "tiered memory", "kv cache"),
    }
    return any(keyword in text for keyword in keyword_groups.get(technology, (technology.lower(),)))


def extract_published_date(html: str) -> str | None:
    """페이지 메타데이터와 JSON-LD에서 공개일을 찾아 ISO 날짜로 반환합니다."""
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    for tag in soup.select("meta[content]"):
        key = (tag.get("property") or tag.get("name") or "").lower()
        if any(token in key for token in ("article:published_time", "datepublished", "pubdate", "date")):
            candidates.append(tag.get("content", ""))

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or script.get_text())
        except (TypeError, json.JSONDecodeError):
            continue
        records = data if isinstance(data, list) else [data]
        candidates.extend(
            record.get("datePublished", "")
            for record in records
            if isinstance(record, dict)
        )

    for value in candidates:
        match = re.search(r"\d{4}-\d{2}-\d{2}", value or "")
        if match:
            return match.group(0)
    return None


def fetch_web_text(url: str, max_chars: int = 8000) -> str:
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 KV-Cache-Research/1.0"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return normalize_text(soup.get_text("\n"))[:max_chars]
    except Exception:
        return ""


def fetch_web_metadata(url: str) -> str | None:
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 KV-Cache-Research/1.0"},
        )
        response.raise_for_status()
        return extract_published_date(response.text)
    except Exception:
        return None


def duckduckgo_html_search(query: str, max_results: int = 4) -> list[dict]:
    """DDGS 검색 공급자가 실패할 경우 사용하는 무료 HTML 검색 대체 경로입니다."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    response = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 KV-Cache-Research/1.0"},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    results = []
    for block in soup.select(".result"):
        link = block.select_one(".result__a")
        if not link:
            continue
        snippet = block.select_one(".result__snippet")
        results.append({
            "title": link.get_text(" ", strip=True),
            "href": link.get("href"),
            "body": snippet.get_text(" ", strip=True) if snippet else "",
        })
        if len(results) >= max_results:
            break
    return results


def tavily_search(query: str, max_results: int = 4) -> list[dict]:
    """Tavily 검색 결과를 공통 결과 형식으로 변환합니다."""
    if tavily_client is None:
        return []

    response = tavily_client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
    )
    return [
        {
            "title": item.get("title", "Untitled"),
            "url": item.get("url"),
            "body": item.get("content", ""),
            "score": item.get("score"),
            "provider": "tavily",
            # Tavily가 발행일을 제공하는 경우에만 채워짐 (일반 웹페이지는 비어있을 수 있음, 임의 생성 안 함)
            "published_date": item.get("published_date"),
        }
        for item in response.get("results", [])
    ]


def web_search(query: str, agent: str, technology: str, max_results: int = 4) -> list[dict]:
    evidence = []
    results = []

    if tavily_client is not None:
        try:
            results = tavily_search(query, max_results=max_results)
        except Exception as tavily_error:
            print(f"[WEB] Tavily 실패, DuckDuckGo로 전환: {type(tavily_error).__name__}: {tavily_error}")

    if not results:
        try:
            # 자동 backend가 Startpage를 선택하지 않도록 DuckDuckGo를 우선 지정합니다.
            results = list(DDGS().text(query, max_results=max_results, backend="duckduckgo"))
        except Exception as first_error:
            print(f"[WEB] DDGS 실패, HTML 검색으로 전환: {type(first_error).__name__}")
            try:
                results = duckduckgo_html_search(query, max_results=max_results)
            except Exception as second_error:
                print(f"[WEB] 대체 검색도 실패: {type(second_error).__name__}: {second_error}")
                return []

    for item in results:
        url = item.get("href") or item.get("url")
        title = item.get("title", "Untitled")
        snippet = item.get("body", "")
        # Tavily는 본문 관련 내용을 content로 돌려주므로 페이지를 다시 요청하지 않습니다.
        is_tavily = item.get("provider") == "tavily"
        page_text = fetch_web_text(url) if url and FETCH_WEB_FULL_TEXT and not is_tavily else ""
        evidence_text = page_text or snippet
        if not url or not is_relevant_result(title, evidence_text, url, technology):
            continue
        published_date = item.get("published_date") or item.get("date")
        if not published_date and url:
            published_date = fetch_web_metadata(url)
        # 동일 URL이 여러 Agent·질의에서 발견되어도 하나의 웹 근거로 취급합니다.
        source_key = url or f"{agent}|{query}"
        evidence_id = make_evidence_id("web", source_key)

        evidence.append(Evidence(
            evidence_id=evidence_id,
            agent=agent,
            technology=technology,
            source_type="web",
            title=title,
            claim=query,
            evidence_text=evidence_text[:8000],
            url=url,
            retrieval_score=item.get("score"),
            published_at=published_date,
        ).model_dump())

    return evidence
