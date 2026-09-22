"""초기화 노드: 비교 대상 기술 2종 확정. (담당: 공통·통합)"""
from __future__ import annotations

from state import AgentState


def technology_selection_agent(state: AgentState) -> dict:
    return {
        "selected_technologies": {
            "software": {"name": "DeepSeek-V2 MLA", "approach": "모델 구조 기반 KV 압축"},
            "hardware": {"name": "ITME", "approach": "CXL-Hybrid 메모리 확장"},
        },
        "selection_reason": (
            "동일한 KV cache 병목을 모델 구조와 메모리 시스템이라는 서로 다른 계층에서 "
            "해결하므로 경쟁 관계와 보완 가능성을 함께 평가할 수 있다."
        ),
        "domain": "데이터센터·클라우드 LLM 서빙",
        "retry_count": 0,
    }
