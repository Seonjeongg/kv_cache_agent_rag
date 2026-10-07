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
from pathlib import Path

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
from graph import build_graph, initial_state, quality_evaluator_node, warning_node
from agents.synthesis import report_generation_agent
from langgraph.graph import END, START, StateGraph
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


def validate_report(
    report: str,
    references: list[dict] | None = None,
    citation_map: dict[str, list[str]] | None = None,
) -> None:
    missing = [heading for heading in REQUIRED_REPORT_HEADINGS if heading not in report]
    if missing:
        raise ValueError(f"보고서 필수 목차 누락: {missing}")
    reference_index = report.rfind("# REFERENCE")
    if reference_index < report.find("# SUMMARY"):
        raise ValueError("REFERENCE는 보고서 마지막에 있어야 합니다.")
    trailing_headings = re.findall(r"^#\s+.+$", report[reference_index:], flags=re.MULTILINE)
    if len(trailing_headings) != 1:
        raise ValueError("REFERENCE 뒤에 다른 보고서 제목이 있거나 REFERENCE가 중복됩니다.")
    if not re.search(r"^- \[\d+\]", report[reference_index:], flags=re.MULTILINE):
        raise ValueError("REFERENCE 항목이 비어 있습니다.")
    body = report[:reference_index]
    reference_section = report[reference_index:]
    if re.search(r"(?:rag|web)-[0-9a-f]{12}", body):
        raise ValueError("최종 보고서 본문에 내부 Evidence ID가 남아 있습니다.")
    body_numbers = set(re.findall(r"\[(\d+)\]", body))
    reference_numbers = set(re.findall(r"^- \[(\d+)\]", reference_section, flags=re.MULTILINE))
    missing_numbers = body_numbers - reference_numbers
    if missing_numbers:
        raise ValueError(f"본문 인용이 REFERENCE에 없습니다: {sorted(missing_numbers, key=int)}")
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
        mapped_ids = {
            evidence_id
            for number in body_numbers
            for evidence_id in (citation_map or {}).get(number, [])
        }
        missing_ids = sorted(mapped_ids - available_ids)
        if missing_ids:
            raise ValueError(f"citation_map의 Evidence가 State references에 없습니다: {missing_ids[:10]}")
    if citation_map is not None:
        map_numbers = set(citation_map)
        if reference_numbers != map_numbers:
            raise ValueError(
                f"REFERENCE 번호와 citation_map 번호가 일치하지 않습니다: "
                f"reference={sorted(reference_numbers, key=int)}, map={sorted(map_numbers, key=int)}"
            )
        unmapped_body = body_numbers - map_numbers
        if unmapped_body:
            raise ValueError(f"본문 인용이 citation_map에 없습니다: {sorted(unmapped_body, key=int)}")


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

    validate_report(result["report"], result.get("references", []), result.get("citation_map", {}))
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


def regenerate_report(state_path: str) -> dict:
    """저장된 조사 결과로 보고서 단계만 재실행한다. 새 조사 실행과 구분한다."""
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    source_trace_id = state.get("trace_id")
    state.update(
        trace_id=str(uuid.uuid4()), report="", citation_map={},
        report_retry_count=0, status="running",
    )
    builder = StateGraph(AgentState)
    builder.add_node("report_generation", report_generation_agent)
    builder.add_node("quality_evaluator", quality_evaluator_node)
    builder.add_node("warning", warning_node)
    builder.add_edge(START, "report_generation")
    builder.add_edge("report_generation", "quality_evaluator")

    def next_step(current):
        verdict = current.get("quality_evaluation", {}).get("result")
        if verdict == "pass":
            return "finish"
        if verdict == "report_retry":
            return "report"
        # 추가 조사가 필요한 결과를 보고서 편집만으로 통과시키지 않는다.
        return "warning"
    builder.add_conditional_edges(
        "quality_evaluator", next_step,
        {"finish": END, "report": "report_generation", "warning": "warning"},
    )
    builder.add_edge("warning", END)
    trace_file = OUTPUT_DIR / f"execution_{state['trace_id']}.jsonl"
    correlation = {
        "trace_id": state["trace_id"], "source_trace_id": source_trace_id,
        "report_only": True,
    }
    result = builder.compile().invoke(state, config={
        "recursion_limit": 20, "run_id": uuid.UUID(state["trace_id"]),
        "run_name": "kv-cache-report-regeneration",
        "metadata": correlation,
        "callbacks": [ExecutionTrace(trace_file, state["trace_id"])],
    })
    result["runtime_metadata"] = {
        **state.get("runtime_metadata", {}), **correlation,
        "local_trace": str(trace_file), "status": result.get("status"),
        "quality_evaluation": result.get("quality_evaluation", {}),
    }
    print("보고서 재생성 결과:", result["quality_evaluation"]["result"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV-cache Agentic RAG 실행")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--request",
        default=None,
        help="조사 관점이 포함된 요청문. 생략하면 네 관점을 모두 조사합니다.",
    )
    mode.add_argument(
        "--report-from-state",
        help="기존 조사 State JSON으로 보고서만 재생성 (전체 실행과 별도 trace)",
    )
    args = parser.parse_args()
    if args.report_from_state:
        pipeline_result = regenerate_report(args.report_from_state)
    else:
        pipeline_result = run_pipeline(args.request)
    save_outputs(pipeline_result)
