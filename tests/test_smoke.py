"""비-LLM 로직에 대한 최소 self-check. pytest 없이 `python tests/test_smoke.py`로 실행."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.synthesis import has_usable_analysis, validation_judge
import evaluate as retriever_evaluation
from agents.domain_evaluation import (
    DOMAIN_CRITERIA,
    build_domain_queries,
    normalize_domain_analysis,
)
from app import validate_report
from evidence import format_references, make_evidence_id
from rag import _rerank, split_text
from graph import build_graph


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


def test_malformed_evidence_id_does_not_retry():
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
    assert result["validation_result"] == "pass_with_limitations"
    assert result["retry_count"] == 0


def test_report_requires_submission_headings():
    report = "\n\n".join([
        "# SUMMARY", "# 1. 분석 배경", "# 2. 기술 선정", "# 3. 기술 개요",
        "# 4. 관점별 평가", "# 6. 시사점", "# 7. 분석의 한계",
        "# REFERENCE", "- [rag-aaaaaaaaaaaa] Paper",
    ])
    validate_report(report)


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
        "__start__", "__end__", "technology_selection", "technical_research",
        "market_evaluation", "stakeholder_evaluation", "domain_evaluation",
        "synthesis", "validation", "report_generation",
    }
    assert expected.issubset(node_names), node_names


def test_evaluate_retriever_reports_question_level_details():
    def fake_retrieve(*args, **kwargs):
        return [
            {"chunk_id": "wrong", "page": 1},
            {"chunk_id": "target", "page": 2},
        ]

    original_retrieve = retriever_evaluation.retrieve
    retriever_evaluation.retrieve = fake_retrieve
    try:
        result = retriever_evaluation.evaluate_retriever(
            [
                {
                    "id": "case-1",
                    "question": "질문",
                    "technology": "DeepSeek-V2 MLA",
                    "ground_truth_chunk_ids": ["target"],
                }
            ],
            k=2,
            candidate_k=4,
        )
    finally:
        retriever_evaluation.retrieve = original_retrieve

    assert result["Hit@2"] == 1.0
    assert result["MRR"] == 0.5
    assert result["details"][0]["matched_chunk_id"] == "target"
    assert result["details"][0]["retrieved_chunk_ids"] == ["wrong", "target"]


def test_domain_queries_are_separated_by_technology_and_criterion():
    queries = build_domain_queries()
    assert len(queries) == len(DOMAIN_CRITERIA) * 2
    assert len({item["query"] for item in queries}) == len(queries)
    assert {item["side"] for item in queries} == {"software", "hardware"}
    assert {item["criterion_key"] for item in queries} == {
        item["key"] for item in DOMAIN_CRITERIA
    }
    for query in queries:
        criterion = next(
            item for item in DOMAIN_CRITERIA
            if item["key"] == query["criterion_key"]
        )
        assert criterion["query_term"] in query["query"]


def test_domain_analysis_has_a_stable_schema_for_all_criteria():
    normalized = normalize_domain_analysis(
        {
            "criteria": {
                "hbm_memory_usage": {
                    "finding": "HBM 절감 근거",
                    "evidence_ids": ["evidence-1"],
                }
            }
        },
        "DeepSeek-V2 MLA",
    )

    assert set(normalized["criteria"]) == {
        item["key"] for item in DOMAIN_CRITERIA
    }
    assert normalized["criteria"]["hbm_memory_usage"]["finding"] == "HBM 절감 근거"
    assert normalized["criteria"]["throughput"]["finding"] == "공개 정보 부족"


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"\n{len(tests)} smoke tests passed")
