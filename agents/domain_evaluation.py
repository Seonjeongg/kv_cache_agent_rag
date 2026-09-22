"""도메인 평가 Agent: 데이터센터·클라우드 LLM 서빙 적용성."""
from __future__ import annotations

import json
from typing import Any

from config import AGENT_RAG_TOP_K
from evidence import compact_evidence, rag_evidence
from llm import ask_json
from state import AgentState

DEFAULT_TECHNOLOGIES = {
    "software": "DeepSeek-V2 MLA",
    "hardware": "ITME",
}

# 현용찬: 설계서의 '도메인 적용 관점'을 안정적인 키로 정의했다.
# 각 기준은 별도 검색 질의로 사용하며, FAST_MODE에서도 기준을 하나로 합치지 않는다.
DOMAIN_CRITERIA = (
    {
        "key": "hbm_memory_usage",
        "label": "HBM 사용량",
        "query_term": "GPU HBM memory usage and KV cache capacity",
    },
    {
        "key": "long_context_scalability",
        "label": "장문맥 확장성",
        "query_term": "long-context scalability",
    },
    {
        "key": "concurrent_user_capacity",
        "label": "동시 사용자 수용 능력",
        "query_term": "concurrent user capacity and request serving",
    },
    {
        "key": "time_to_first_token",
        "label": "첫 토큰 응답 시간",
        "query_term": "time to first token and latency",
    },
    {
        "key": "throughput",
        "label": "처리량",
        "query_term": "LLM inference throughput",
    },
    {
        "key": "accuracy_impact",
        "label": "정확도 영향",
        "query_term": "accuracy or quality impact",
    },
    {
        "key": "data_transfer_latency",
        "label": "데이터 전송 지연",
        "query_term": "data movement and transfer latency",
    },
    {
        "key": "infrastructure_requirements",
        "label": "신규 인프라 요구",
        "query_term": "additional infrastructure requirements",
    },
    {
        "key": "serving_system_compatibility",
        "label": "서빙 시스템 호환성",
        "query_term": "LLM serving system compatibility",
    },
    {
        "key": "operational_complexity_cost",
        "label": "운영 복잡도와 비용",
        "query_term": "operational complexity and cost",
    },
)


def _technology_map(state: AgentState) -> dict[str, str]:
    """State에 저장된 선정 기술을 사용하되, 구형 State도 안전하게 지원한다."""
    selected = state.get("selected_technologies", {})
    if not isinstance(selected, dict):
        selected = {}

    technologies = {}
    for side, default_name in DEFAULT_TECHNOLOGIES.items():
        selected_item = selected.get(side, {})
        if isinstance(selected_item, dict):
            technologies[side] = selected_item.get("name") or default_name
        else:
            technologies[side] = str(selected_item or default_name)
    return technologies


def build_domain_queries(
    technologies: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """기술별·평가기준별 RAG 질의를 만든다."""
    technology_map = technologies or DEFAULT_TECHNOLOGIES
    queries = []
    for side, technology in technology_map.items():
        for criterion in DOMAIN_CRITERIA:
            queries.append(
                {
                    "side": side,
                    "technology": technology,
                    "criterion_key": criterion["key"],
                    "query": (
                        f"{technology} impact on {criterion['query_term']} "
                        "in datacenter cloud LLM serving"
                    ),
                }
            )
    return queries


def _criterion_schema() -> dict[str, dict[str, Any]]:
    return {
        criterion["key"]: {
            "finding": "",
            "constraint": "",
            "conditions": "",
            "evidence_ids": [],
        }
        for criterion in DOMAIN_CRITERIA
    }


def _as_evidence_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


def normalize_domain_analysis(raw: Any, technology: str) -> dict[str, Any]:
    """LLM 응답을 State에 넣을 고정된 도메인 분석 Schema로 정규화한다."""
    raw_dict = raw if isinstance(raw, dict) else {}
    source = raw_dict.get("criteria", raw_dict)
    if not isinstance(source, dict):
        source = {}

    criteria = {}
    for criterion in DOMAIN_CRITERIA:
        key = criterion["key"]
        value = source.get(key, {})
        if not isinstance(value, dict):
            value = {"finding": str(value)}

        criteria[key] = {
            "label": criterion["label"],
            "finding": value.get("finding") or "공개 정보 부족",
            "constraint": (
                value.get("constraint")
                or value.get("limitation")
                or "공개 정보 부족"
            ),
            "conditions": (
                value.get("conditions")
                or value.get("comparison_conditions")
                or "공개 정보 부족"
            ),
            "evidence_ids": _as_evidence_ids(value.get("evidence_ids")),
        }

    return {
        "technology": technology,
        "criteria": criteria,
    }


def _analysis_prompt(
    technology: str,
    domain: str,
    evidence: list[dict[str, Any]],
) -> str:
    schema = json.dumps(_criterion_schema(), ensure_ascii=False, indent=2)
    criteria = json.dumps(
        [
            {"key": item["key"], "label": item["label"]}
            for item in DOMAIN_CRITERIA
        ],
        ensure_ascii=False,
        indent=2,
    )
    return f"""
당신은 {domain} 분야의 도메인 평가 전문가입니다.
이번 분석 대상은 반드시 {technology} 하나뿐입니다.
다른 기술의 내용을 이 분석에 섞거나, 다른 기술과 직접 비교하지 마세요.

다음 10개 기준을 모두 평가하세요.
{criteria}

각 기준마다 다음을 작성하세요.
- finding: 확인된 효과 또는 관찰
- constraint: 한계, 위험, 도입 장벽
- conditions: 수치가 보고된 실험·적용 조건. 확인되지 않으면 공개 정보 부족
- evidence_ids: 아래 근거에 실제로 존재하는 evidence_id만 사용

반드시 다음 JSON 구조를 지키세요.
{schema}

정확한 근거가 없으면 추측하지 말고 해당 필드에 "공개 정보 부족"을 적으세요.
근거의 실험 조건이 다르면 성능 수치를 우열 비교에 사용하지 말고 조건 차이를 기록하세요.

{technology} 전용 근거:
{compact_evidence(evidence)}
"""


def _combine_analyses(
    analyses: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """두 기술의 개별 분석을 동일 기준의 비교 가능한 State 구조로 묶는다."""
    criteria = {}
    for criterion in DOMAIN_CRITERIA:
        key = criterion["key"]
        criteria[key] = {
            "label": criterion["label"],
            "software": analyses["software"]["criteria"][key],
            "hardware": analyses["hardware"]["criteria"][key],
        }

    return {
        "criteria": criteria,
        "software": analyses["software"],
        "hardware": analyses["hardware"],
        "comparison_principle": (
            "서로 다른 실험·적용 조건의 수치를 단순 비교하지 않고, "
            "조건 차이와 공개 정보 부족을 함께 기록한다."
        ),
    }


def domain_evaluation_agent(state: AgentState) -> dict:
    """도메인 기준별 검색과 기술별 분석을 수행한다."""
    agent_name = "domain_evaluation"
    print("[4/6] 도메인 평가 Agent 시작")

    try:
        technologies = _technology_map(state)
        domain = state.get("domain", "데이터센터·클라우드 LLM 서빙")
        evidence_by_side = {side: [] for side in technologies}

        # 현용찬: 기존 FAST_MODE의 통합 질의를 제거했다.
        # 기준별 질의를 유지해야 HBM, 지연, 처리량 등 어떤 기준의 근거가
        # 부족한지 확인할 수 있고, 도메인 평가 결과를 설계서의 State 키와
        # 일대일로 대응시킬 수 있다.
        for query_item in build_domain_queries(technologies):
            evidence_by_side[query_item["side"]].extend(
                rag_evidence(
                    query_item["query"],
                    query_item["technology"],
                    agent_name,
                    top_k=AGENT_RAG_TOP_K,
                )
            )

        analyses = {}
        for side, technology in technologies.items():
            # 현용찬: MLA와 ITME를 별도 프롬프트로 분석한다.
            # 한 프롬프트에 두 기술의 근거를 넣으면 서로 다른 기술의
            # 원리·수치·한계가 섞이는 오류가 발생할 수 있기 때문이다.
            raw_analysis = ask_json(
                "당신은 근거 중심의 데이터센터 LLM 서빙 평가 전문가입니다.",
                _analysis_prompt(
                    technology,
                    domain,
                    evidence_by_side[side],
                ),
            )
            if isinstance(raw_analysis, dict) and isinstance(
                raw_analysis.get("domain_analysis"), dict
            ):
                raw_analysis = raw_analysis["domain_analysis"]
            analyses[side] = normalize_domain_analysis(raw_analysis, technology)

        domain_analysis = _combine_analyses(analyses)
        all_evidence = [
            item
            for side in technologies
            for item in evidence_by_side[side]
        ]
        print(f"[4/6] 도메인 평가 완료 - 근거 {len(all_evidence)}건")
        return {
            "domain_analysis": domain_analysis,
            "references": all_evidence,
        }
    except Exception as error:
        return {
            "domain_analysis": {},
            "errors": [f"{agent_name}: {error}"],
        }
