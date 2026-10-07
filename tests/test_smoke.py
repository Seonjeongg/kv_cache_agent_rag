"""비-LLM 로직에 대한 최소 self-check. pytest 없이 `python tests/test_smoke.py`로 실행."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.synthesis import (
    ensure_section_depth,
    exclude_unregistered_evidence,
    has_usable_analysis,
    validation_judge,
)
import agents.synthesis as synthesis_module
import graph as graph_module
from app import validate_report
from evidence import format_references, make_evidence_id
from rag import _rerank, split_text
from graph import (
    NEUTRALITY_PATTERNS,
    _remove_unregistered_ids,
    build_graph,
    fan_out_or_warn,
    initial_state,
    _task_plan,
    quality_evaluator_node,
    route_after_quality,
    route_after_validation,
)


def test_split_text_respects_max_chars():
    text = "가" * 3000
    chunks = split_text(text, max_chars=1400, overlap=150)
    assert all(len(c) <= 1400 for c in chunks)
    assert "".join(chunks).replace("", "") != ""  # non-empty


def test_split_text_short_text_returns_single_chunk():
    assert split_text("짧은 문장") == ["짧은 문장"]


def test_rerank_preserves_relevant_lexical_match():
    results = [
        {"chunk_id": "semantic", "text": "attention memory architecture", "similarity": 0.9},
        {"chunk_id": "exact", "text": "reduce KV cache key value elements", "similarity": 0.8},
    ]
    reranked = _rerank("reduce KV cache", results, top_k=2)
    assert reranked[0]["chunk_id"] == "exact"


def test_make_evidence_id_is_deterministic():
    a = make_evidence_id("rag", "agent|chunk|query")
    b = make_evidence_id("rag", "agent|chunk|query")
    assert a == b
    assert a.startswith("rag-")


def test_validation_rejects_parse_error_result():
    assert not has_usable_analysis({"parse_error": True})
    assert not has_usable_analysis({"software": {"principle": ""}})


def test_empty_reference_filter_stays_empty():
    references = [{"evidence_id": "rag-aaaaaaaaaaaa", "source_type": "paper", "title": "x"}]
    assert format_references(references, used_ids=set()) == "- 실제 활용 자료 없음"


def test_web_reference_shows_unknown_date():
    references = [{
        "evidence_id": "web-aaaaaaaaaaaa",
        "source_type": "web",
        "title": "Example",
        "url": "https://example.com/article",
    }]
    assert "example.com(날짜 미상). Example" in format_references(references)


def test_reference_display_deduplicates_same_source():
    references = [
        {"evidence_id": "rag-aaaaaaaaaaaa", "source_type": "paper", "file_name": "paper.pdf", "title": "Paper", "page": 1},
        {"evidence_id": "rag-bbbbbbbbbbbb", "source_type": "paper", "file_name": "paper.pdf", "title": "Paper", "page": 2},
        {"evidence_id": "web-cccccccccccc", "source_type": "web", "url": "https://example.com/a", "title": "A"},
        {"evidence_id": "web-dddddddddddd", "source_type": "web", "url": "https://example.com/a", "title": "A"},
    ]
    rendered = format_references(references)
    assert rendered.count("paper.pdf") == 1
    assert "pp.1, 2" in rendered
    assert "[rag-aaaaaaaaaaaa] [rag-bbbbbbbbbbbb]" in rendered
    assert rendered.count("https://example.com/a") == 1


def test_malformed_evidence_id_is_ignored():
    state = {
        "technical_analysis": {"software": {"principle": "ok"}},
        "trl_analysis": {"software": {"reason": "ok"}},
        "market_analysis": {"software": {"evidence_ids": ["web-ITME-1"]}},
        "stakeholder_analysis": {"cloud_provider": {"expectation": "ok"}},
        "domain_analysis": {"comparison": "ok"},
        "synthesis": {"summary": "ok"},
        "references": [{"evidence_id": "rag-aaaaaaaaaaaa"}],
        "retry_count": 0,
    }
    result = validation_judge(state)
    assert result["validation_result"] == "pass"
    assert result["retry_count"] == 0


def test_report_requires_submission_headings():
    long_section = "가" * 450 + "\n\n" + "나" * 450
    report = "\n\n".join([
        "# SUMMARY", "# 1. 분석 배경", "# 2. 기술 선정", "# 3. 기술 개요",
        "# 4. 관점별 평가", "# 5. 관점별 비교 및 상충 지점", f"# 6. 시사점\n{long_section}", f"# 7. 분석의 한계\n{long_section}",
        "# REFERENCE", "- [rag-aaaaaaaaaaaa] Paper",
    ])
    validate_report(report)


def test_report_rejects_unregistered_evidence_id():
    long_section = "가" * 450 + "\n\n" + "나" * 450
    report = "\n\n".join([
        "# SUMMARY\n주장 [rag-bbbbbbbbbbbb]", "# 1. 분석 배경", "# 2. 기술 선정",
        "# 3. 기술 개요", "# 4. 관점별 평가", "# 5. 관점별 비교 및 상충 지점", f"# 6. 시사점\n{long_section}", f"# 7. 분석의 한계\n{long_section}",
        "# REFERENCE", "- [rag-aaaaaaaaaaaa] Paper",
    ])
    try:
        validate_report(report, [{"evidence_id": "rag-aaaaaaaaaaaa"}])
    except ValueError as error:
        assert "본문에 인용된 Evidence" in str(error)
        return
    raise AssertionError("REFERENCES에 없는 Evidence가 통과했습니다.")


def test_unknown_evidence_retries_only_the_own_worker():
    state = {
        "technical_analysis": {"software": {"evidence_ids": ["rag-aaaaaaaaaaaa"]}},
        "trl_analysis": {"software": {"evidence_ids": ["rag-aaaaaaaaaaaa"]}},
        "market_analysis": {"software": {"evidence_ids": ["rag-bbbbbbbbbbbb"]}},
        "stakeholder_analysis": {"cloud_provider": {"evidence": "ok"}},
        "domain_analysis": {"comparison": "ok"},
        "synthesis": {"summary": "ok", "evidence_ids": ["rag-aaaaaaaaaaaa"]},
        "references": [{"evidence_id": "rag-aaaaaaaaaaaa"}],
        "retry_count": 0,
    }
    previous_max_retries = synthesis_module.MAX_RETRIES
    synthesis_module.MAX_RETRIES = 1
    try:
        result = validation_judge(state)
    finally:
        synthesis_module.MAX_RETRIES = previous_max_retries
    assert result["validation_result"] == "retry"
    assert result["retry_targets"] == ["market_evaluation"]
    assert route_after_validation(result) == "retry"


def test_report_rejects_empty_references():
    report = "\n\n".join([
        "# SUMMARY", "# 1. 분석 배경", "# 2. 기술 선정", "# 3. 기술 개요",
        "# 4. 관점별 평가", "# 6. 시사점", "# 7. 분석의 한계", "# REFERENCE",
    ])
    try:
        validate_report(report)
    except ValueError:
        return
    raise AssertionError("빈 REFERENCE가 통과했습니다.")


def test_graph_builds_with_all_nodes_wired():
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__", "__end__", "technology_selection", "orchestrator_plan",
        "worker", "synthesis", "validation", "report_generation",
        "quality_evaluator", "warning",
    }
    assert expected.issubset(node_names), node_names


def test_dynamic_plan_changes_with_requested_perspectives():
    state = initial_state()
    state["selected_technologies"] = {
        "software": {"name": "DeepSeek-V2 MLA"},
        "hardware": {"name": "ITME"},
    }
    plans = []
    for request in (
        "시장성만 평가하라.",
        "기술 성숙도와 도메인 적용성을 평가하라.",
        "기술 성숙도, 시장성과 도메인 적용성을 평가하라.",
        "기술 성숙도, 시장성, 이해관계자, 도메인 적용성을 평가하라.",
    ):
        state["input_request"] = request
        plan = _task_plan(state)
        state["plan"] = plan
        plans.append(plan)
        sends = fan_out_or_warn(state)
        assert len(sends) == len(plan)
        assert {send.node for send in sends} == {"worker"}
    assert [len(plan) for plan in plans] == [1, 2, 3, 4]
    assert all(plan[0].get("subtask_id") for plan in plans)


def test_retry_plan_only_contains_requested_targets():
    state = initial_state("전체 관점으로 평가하라.")
    state["retry_targets"] = ["market_evaluation"]
    plan = _task_plan(state)
    assert [task["task_id"] for task in plan] == ["market_evaluation"]


def test_initial_state_accepts_runtime_request():
    state = initial_state("시장성과 도메인 적용성만 비교하라.")
    assert state["input_request"] == "시장성과 도메인 적용성만 비교하라."


def test_quality_evaluator_requires_grounded_and_diverse_evidence():
    state = {
        "report": "# SUMMARY\n근거 요약 [rag-aaaaaaaaaaaa]\n# 1. 분석 배경\n# 2. 기술 선정\n# 3. 기술 개요\n# 4. 관점별 평가\n# 6. 시사점\n# 7. 분석의 한계\n# REFERENCE",
        "references": [{"evidence_id": "rag-aaaaaaaaaaaa", "file_name": "same.pdf", "title": "same"}],
        "technical_analysis": {"software": {"evidence_ids": ["rag-aaaaaaaaaaaa"]}},
        "trl_analysis": {"software": {"evidence_ids": ["rag-aaaaaaaaaaaa"]}},
        "market_analysis": {"software": {"evidence_ids": ["rag-aaaaaaaaaaaa"]}},
        "stakeholder_analysis": {"software": {"evidence_ids": ["rag-aaaaaaaaaaaa"]}},
        "domain_analysis": {"software": {"evidence_ids": ["rag-aaaaaaaaaaaa"]}},
        "synthesis": {"summary": "근거 기반 요약"},
        "retry_count": 0,
    }
    result = quality_evaluator_node(state)
    assert result["quality_evaluation"]["criteria"]["groundedness"] is False
    state["report"] += "\n- [rag-aaaaaaaaaaaa] same.pdf"
    result = quality_evaluator_node(state)
    assert result["quality_evaluation"]["criteria"]["groundedness"] is True
    assert result["quality_evaluation"]["criteria"]["bias_control"] is False
    assert "bias_control" in result["quality_evaluation"]["issues"]


def test_quality_evaluator_rejects_empty_grounding_and_recommendation_language():
    state = {
        "report": "# SUMMARY\nMLA가 가장 우수하다.\n# 1. 분석 배경\n# 2. 기술 선정\n# 3. 기술 개요\n# 4. 관점별 평가\n# 6. 시사점\n# 7. 분석의 한계\n# REFERENCE",
        "references": [{"evidence_id": "rag-aaaaaaaaaaaa", "file_name": "paper.pdf", "title": "paper"}],
        "technical_analysis": {"software": {"principle": "근거 없는 주장"}},
        "trl_analysis": {"software": {"reason": "근거 없는 주장"}},
        "market_analysis": {"software": {"analysis": "근거 없는 주장"}},
        "stakeholder_analysis": {"software": {"analysis": "근거 없는 주장"}},
        "domain_analysis": {"software": {"analysis": "근거 없는 주장"}},
        "synthesis": {"summary": "근거 없는 요약"},
        "retry_count": 0,
    }
    result = quality_evaluator_node(state)
    criteria = result["quality_evaluation"]["criteria"]
    assert criteria["groundedness"] is False
    assert criteria["neutrality"] is False


def test_neutrality_does_not_flag_optimization_as_recommendation():
    import re

    report = "MLA 기반으로 긴 문맥 처리에 최적화되어 있다."
    assert not any(re.search(pattern, report, flags=re.IGNORECASE) for pattern in NEUTRALITY_PATTERNS)

    neutral_comparison = "비교의 핵심은 승자를 정하는 것이 아니라 조건별 병목을 확인하는 것이다."
    assert not any(re.search(pattern, neutral_comparison, flags=re.IGNORECASE) for pattern in NEUTRALITY_PATTERNS)

    winner_claim = "MLA가 승자다."
    assert any(re.search(pattern, winner_claim, flags=re.IGNORECASE) for pattern in NEUTRALITY_PATTERNS)


def test_neutrality_flags_recommendation_like_market_language():
    import re

    report = "최적의 조합이며 상용화가 가속화될 수 있다."
    assert any(re.search(pattern, report, flags=re.IGNORECASE) for pattern in NEUTRALITY_PATTERNS)


def test_validation_can_retry_synthesis_without_unknown_worker_task():
    assert route_after_validation({"validation_result": "retry", "retry_targets": ["synthesis"]}) == "synthesis"


def test_quality_report_retry_routes_to_report_generation():
    state = {"quality_evaluation": {"result": "report_retry"}}
    assert route_after_quality(state) == "report_retry"


def test_report_retry_has_a_separate_budget_after_worker_retries():
    state = {
        "report": "# SUMMARY\n잘못된 인용 [rag-bbbbbbbbbbbb]\n# 4. 관점별 평가\n## 4.2 시장성\n시장 근거 [web-aaaaaaaaaaaa] [web-bbbbbbbbbbbb]\n# REFERENCE",
        "references": [
            {"evidence_id": "web-aaaaaaaaaaaa", "source_type": "web", "url": "https://a.example", "title": "A"},
            {"evidence_id": "web-bbbbbbbbbbbb", "source_type": "web", "url": "https://b.example", "title": "B"},
        ],
        "required_task_ids": ["market_evaluation"],
        "market_analysis": {"analysis": "ok"},
        "synthesis": {"summary": "ok"},
        "retry_count": 2,
        "report_retry_count": 0,
    }
    previous_max_retries = graph_module.MAX_RETRIES
    graph_module.MAX_RETRIES = 2
    try:
        result = quality_evaluator_node(state)
    finally:
        graph_module.MAX_RETRIES = previous_max_retries
    assert result["quality_evaluation"]["result"] == "report_retry"
    assert result["report_retry_count"] == 1
    assert route_after_quality(result) == "report_retry"


def test_unregistered_evidence_paragraph_is_excluded():
    report = "확인된 주장 [rag-aaaaaaaaaaaa]\n\n등록되지 않은 주장 [rag-bbbbbbbbbbbb]"
    safe = exclude_unregistered_evidence(report, {"rag-aaaaaaaaaaaa"})
    assert "rag-bbbbbbbbbbbb" not in safe
    assert "출처와 연결할 수 없어" in safe
    assert "등록되지 않은 주장" not in safe


def test_unregistered_claim_is_removed_without_losing_verified_sentence():
    text = "확인한 원리 [rag-aaaaaaaaaaaa]. 비용이 99% 감소했다 [rag-bbbbbbbbbbbb]."
    safe = exclude_unregistered_evidence(text, {"rag-aaaaaaaaaaaa"})
    assert "확인한 원리" in safe
    assert "99%" not in safe


def test_validation_does_not_add_unrequested_workers():
    result = validation_judge({
        "required_task_ids": ["market_evaluation"],
        "market_analysis": {"evidence_ids": ["web-aaaaaaaaaaaa"], "analysis": "시장 자료"},
        "synthesis": {"summary": "정리"},
        "references": [{"evidence_id": "web-aaaaaaaaaaaa"}],
    })
    assert result["validation_result"] == "pass"
    assert result["retry_targets"] == []


def test_reference_reducer_preserves_ids_for_same_url():
    from state import merge_references
    common = {"source_type": "web", "url": "https://example.com/a"}
    references = merge_references(
        [{**common, "evidence_id": "web-aaaaaaaaaaaa"}],
        [{**common, "evidence_id": "web-bbbbbbbbbbbb"}],
    )
    assert {item["evidence_id"] for item in references} == {"web-aaaaaaaaaaaa", "web-bbbbbbbbbbbb"}


def test_short_implications_and_limitations_get_safe_depth_fallback():
    for name in ("implications", "limitations"):
        section = ensure_section_depth("짧은 초안", name, 900)
        assert len(section) >= 900
        assert section.count("\n\n") >= 1


def test_short_comparison_gets_safe_depth_fallback():
    section = ensure_section_depth("짧은 초안", "comparison_conflicts", 650)
    assert len(section) >= 650
    assert section.count("\n\n") >= 1


def test_worker_filters_only_unregistered_evidence_ids():
    result = _remove_unregistered_ids(
        {"evidence_ids": ["rag-aaaaaaaaaaaa", "rag-bbbbbbbbbbbb"]},
        {"rag-aaaaaaaaaaaa"},
    )
    assert result["evidence_ids"] == ["rag-aaaaaaaaaaaa"]


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"\n{len(tests)} smoke tests passed")
