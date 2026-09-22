"""비-LLM 로직에 대한 최소 self-check. pytest 없이 `python tests/test_smoke.py`로 실행."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evidence import make_evidence_id
from rag import split_text
from graph import build_graph


def test_split_text_respects_max_chars():
    text = "가" * 3000
    chunks = split_text(text, max_chars=1400, overlap=150)
    assert all(len(c) <= 1400 for c in chunks)
    assert "".join(chunks).replace("", "") != ""  # non-empty


def test_split_text_short_text_returns_single_chunk():
    assert split_text("짧은 문장") == ["짧은 문장"]


def test_make_evidence_id_is_deterministic():
    a = make_evidence_id("rag", "agent|chunk|query")
    b = make_evidence_id("rag", "agent|chunk|query")
    assert a == b
    assert a.startswith("rag-")


def test_graph_builds_with_all_nodes_wired():
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    expected = {
        "__start__", "__end__", "technology_selection", "technical_research",
        "market_evaluation", "stakeholder_evaluation", "domain_evaluation",
        "synthesis", "validation", "report_generation",
    }
    assert expected.issubset(node_names), node_names


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"OK: {test.__name__}")
    print(f"\n{len(tests)} smoke tests passed")
