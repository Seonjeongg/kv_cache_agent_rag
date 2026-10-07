"""두 관점만 요청해 실제 Worker 수가 줄어드는지 확인합니다. 제출 보고서는 생성하지 않습니다."""
from __future__ import annotations

import json
import uuid
from config import OUTPUT_DIR
from graph import build_graph, initial_state
from rag import get_collection
from tracing import ExecutionTrace


if __name__ == "__main__":
    get_collection()
    state = initial_state("기술 성숙도와 도메인 적용성만 평가하라.")
    trace_id = state["trace_id"]
    result = build_graph().invoke(
        state,
        config={
            "run_id": uuid.UUID(trace_id),
            "run_name": "kv-cache-two-worker-demo",
            "metadata": {"trace_id": trace_id, "purpose": "dynamic-fan-out-demo"},
            "callbacks": [ExecutionTrace(OUTPUT_DIR / f"execution_{trace_id}.jsonl", trace_id)],
        },
        interrupt_before=["synthesis"],
    )
    tasks = [task["task_id"] for task in result["plan"]]
    assert tasks == ["technical_research", "domain_evaluation"], tasks
    path = OUTPUT_DIR / "fanout_demo.json"
    path.write_text(json.dumps({
        "trace_id": trace_id,
        "purpose": "두 Worker 실행 후 의도적으로 정지한 동적 fan-out 실증",
        "plan": result["plan"],
        "task_results": result["task_results"],
        "errors": result.get("errors", []),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("실행 Worker:", tasks)
    print("trace_id:", trace_id)
    print("실증 결과:", path)
