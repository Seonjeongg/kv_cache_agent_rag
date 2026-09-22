"""전체 파이프라인 실행 진입점: 색인 -> Graph 실행 -> 보고서 저장."""
from __future__ import annotations

import json
from datetime import datetime

import markdown as markdown_lib

from config import OUTPUT_DIR, PROJECT_DIR
from graph import build_graph
from rag import build_index, download_papers, load_and_chunk_papers
from state import AgentState


def run_pipeline() -> dict:
    download_papers()
    chunks = load_and_chunk_papers()
    collection = build_index(chunks)
    print(f"청크 수: {len(chunks):,}")
    print(f"Vector DB 수: {collection.count():,}")

    graph = build_graph()
    initial_state: AgentState = {
        "input_request": (
            "DeepSeek-V2 MLA와 ITME를 데이터센터·클라우드 LLM 서빙 환경에서 "
            "TRL, 시장성, 이해관계자, 도메인 관점으로 중립적으로 비교 평가하라."
        ),
        "references": [],
        "errors": [],
    }
    result = graph.invoke(initial_state, config={"recursion_limit": 40})

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

    markdown_path.write_text(result["report"], encoding="utf-8")
    html_body = markdown_lib.markdown(result["report"], extensions=["tables", "fenced_code"])
    html_document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo',sans-serif;max-width:900px;margin:40px auto;line-height:1.7;padding:0 24px}} table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ccc;padding:8px}} h1,h2{{margin-top:32px}}</style>
</head><body>{html_body}</body></html>"""
    html_path.write_text(html_document, encoding="utf-8")

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        from weasyprint import HTML
        HTML(string=html_document, base_url=str(PROJECT_DIR)).write_pdf(pdf_path)
        print("PDF:", pdf_path)
    except Exception as error:
        print("PDF 자동 생성 생략:", error)
        print("HTML을 브라우저에서 열어 PDF로 인쇄하세요:", html_path)

    print("Markdown:", markdown_path)
    print("HTML:", html_path)
    print("State JSON:", json_path)
    return {"markdown": markdown_path, "html": html_path, "pdf": pdf_path, "json": json_path}


if __name__ == "__main__":
    pipeline_result = run_pipeline()
    save_outputs(pipeline_result)
