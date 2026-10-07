"""전체 파이프라인 실행 진입점: 색인 -> Graph 실행 -> 보고서 저장."""
from __future__ import annotations

import argparse
import json
import platform
import re
import os
import uuid
from datetime import datetime
from importlib.metadata import version

import markdown as markdown_lib

from config import (
    AGENT_RAG_TOP_K,
    CHUNK_MAX_CHARS,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    FAST_MODE,
    LLM_MODEL,
    OUTPUT_DIR,
    PROJECT_DIR,
    TOP_K,
)
from graph import build_graph, initial_state
from rag import build_index, download_papers, load_and_chunk_papers
from state import AgentState
from tracing import ExecutionTrace


REQUIRED_REPORT_HEADINGS = (
    "# SUMMARY",
    "# 1. 분석 배경",
    "# 2. 기술 선정",
    "# 3. 기술 개요",
    "# 4. 관점별 평가",
    "# 5. 관점별 비교 및 상충 지점",
    "# 6. 시사점",
    "# 7. 분석의 한계",
    "# REFERENCE",
)
REPORT_DEPTH_REQUIREMENTS = {
    "# 6. 시사점": 900,
    "# 7. 분석의 한계": 900,
}


def validate_report(report: str, references: list[dict] | None = None) -> None:
    missing = [heading for heading in REQUIRED_REPORT_HEADINGS if heading not in report]
    if missing:
        raise ValueError(f"보고서 필수 목차 누락: {missing}")
    reference_index = report.rfind("# REFERENCE")
    if reference_index < report.find("# SUMMARY"):
        raise ValueError("REFERENCE는 보고서 마지막에 있어야 합니다.")
    trailing_headings = re.findall(r"^#\s+.+$", report[reference_index:], flags=re.MULTILINE)
    if len(trailing_headings) != 1:
        raise ValueError("REFERENCE 뒤에 다른 보고서 제목이 있거나 REFERENCE가 중복됩니다.")
    if not re.search(r"^- \[[a-z]+-[0-9a-f]{12}\]", report[reference_index:], flags=re.MULTILINE):
        raise ValueError("REFERENCE 항목이 비어 있습니다.")
    for heading, minimum in REPORT_DEPTH_REQUIREMENTS.items():
        match = re.search(rf"^{re.escape(heading)}\s*$", report, flags=re.MULTILINE)
        if not match:
            raise ValueError(f"보고서 절 누락: {heading}")
        section = report[match.end():]
        next_heading = re.search(r"^#\s+", section, flags=re.MULTILINE)
        section = section[:next_heading.start()] if next_heading else section
        if len(section.strip()) < minimum or section.count("\n\n") < 1:
            raise ValueError(f"보고서 절이 너무 짧습니다: {heading} (최소 {minimum}자, 2문단)")
    if references is not None:
        available_ids = {item.get("evidence_id") for item in references}
        used_ids = set(re.findall(r"(?:rag|web)-[0-9a-f]{12}", report[:reference_index]))
        missing_ids = sorted(used_ids - available_ids)
        displayed_ids = set(re.findall(r"(?:rag|web)-[0-9a-f]{12}", report[reference_index:]))
        missing_ids = sorted(set(missing_ids) | (used_ids - displayed_ids))
        if missing_ids:
            raise ValueError(f"본문에 인용된 Evidence가 REFERENCES에 없습니다: {missing_ids[:10]}")


def run_pipeline(input_request: str | None = None) -> dict:
    download_papers()
    chunks = load_and_chunk_papers()
    collection = build_index(chunks)
    print(f"청크 수: {len(chunks):,}")
    print(f"Vector DB 수: {collection.count():,}")

    graph = build_graph()
    if FAST_MODE:
        print("[WARN] FAST_MODE=True: 제출용 보고서는 FAST_MODE=False로 실행하세요.")
    state = initial_state(input_request)
    trace_id = state["trace_id"]
    trace_file = OUTPUT_DIR / f"execution_{trace_id}.jsonl"
    tracing_enabled = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "")).lower() == "true"
    tracing_key = bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    if not (tracing_enabled and tracing_key):
        print("[TRACE] 로컬 실행 기록만 저장합니다. LangSmith 제출 캡처는 별도로 필요합니다.")
    result = graph.invoke(state, config={
        "recursion_limit": 40,
        "run_id": uuid.UUID(trace_id),
        "run_name": "kv-cache-orchestrator",
        "metadata": {"trace_id": trace_id, "fast_mode": FAST_MODE},
        "callbacks": [ExecutionTrace(trace_file, trace_id)],
    })
    result["runtime_metadata"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "openai_package": version("openai"),
        "fast_mode": FAST_MODE,
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_top_k": TOP_K,
        "agent_rag_top_k": AGENT_RAG_TOP_K,
        "trace_id": result.get("trace_id"),
        "status": result.get("status"),
        "step_count": result.get("step_count"),
        "quality_evaluation": result.get("quality_evaluation", {}),
        "local_trace": str(trace_file),
        "langsmith_enabled": tracing_enabled and tracing_key,
    }

    print("검증 결과:", result["validation_result"])
    print("재조사 횟수:", result["retry_count"])
    print("오류:", result.get("errors", []))
    print("Reference 수:", len(result.get("references", [])))
    return result


def save_outputs(result: dict) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_path = OUTPUT_DIR / f"kv_cache_report_{timestamp}.md"
    html_path = OUTPUT_DIR / f"kv_cache_report_{timestamp}.html"
    pdf_path = OUTPUT_DIR / f"kv_cache_report_{timestamp}.pdf"
    json_path = OUTPUT_DIR / f"kv_cache_state_{timestamp}.json"

    validate_report(result["report"], result.get("references", []))
    markdown_path.write_text(result["report"], encoding="utf-8")
    html_body = markdown_lib.markdown(result["report"], extensions=["tables", "fenced_code"])
    html_document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<style>{(PROJECT_DIR / "docs/report.css").read_text(encoding="utf-8")}</style>
</head><body>{html_body}</body></html>"""
    html_path.write_text(html_document, encoding="utf-8")

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    from weasyprint import HTML
    HTML(string=html_document, base_url=str(PROJECT_DIR)).write_pdf(pdf_path)
    import pymupdf
    with pymupdf.open(pdf_path) as document:
        page_count = len(document)
    if page_count > 10:
        raise ValueError(f"제출 보고서가 10장을 초과했습니다: {page_count}장")
    print(f"PDF ({page_count}장):", pdf_path)

    print("Markdown:", markdown_path)
    print("HTML:", html_path)
    print("State JSON:", json_path)
    return {"markdown": markdown_path, "html": html_path, "pdf": pdf_path, "json": json_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV-cache Agentic RAG 실행")
    parser.add_argument(
        "--request",
        default=None,
        help="조사 관점이 포함된 요청문. 생략하면 네 관점을 모두 조사합니다.",
    )
    args = parser.parse_args()
    pipeline_result = run_pipeline(args.request)
    save_outputs(pipeline_result)
