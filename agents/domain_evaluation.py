"""도메인 평가 Agent: 데이터센터·클라우드 서빙 적용성. (담당: 도메인 평가)"""
from __future__ import annotations

import json

from config import AGENT_RAG_TOP_K, FAST_MODE
from evidence import compact_evidence, rag_evidence
from llm import ask_json
from state import AgentState

DOMAIN_CRITERIA = [
    "HBM memory usage",
    "long-context scalability",
    "concurrent user capacity",
    "time to first token",
    "throughput",
    "accuracy impact",
    "data transfer latency",
    "new infrastructure requirements",
    "serving-system compatibility",
    "operational complexity and cost",
]


def domain_evaluation_agent(state: AgentState) -> dict:
    agent_name = "domain_evaluation"
    print("[4/6] 도메인 평가 Agent 시작")
    try:
        evidence = []
        for technology in ["DeepSeek-V2 MLA", "ITME"]:
            criteria_queries = DOMAIN_CRITERIA
            if FAST_MODE:
                criteria_queries = ["; ".join(DOMAIN_CRITERIA)]
            for criterion in criteria_queries:
                query = f"{technology} impact on {criterion} in datacenter cloud LLM serving"
                evidence.extend(rag_evidence(query, technology, agent_name, top_k=AGENT_RAG_TOP_K))

        prompt = f"""
데이터센터·클라우드 LLM 서빙에서 DeepSeek-V2 MLA와 ITME를 동일한 기준으로 평가하세요.
기준: {json.dumps(DOMAIN_CRITERIA, ensure_ascii=False)}
각 기준마다 두 기술의 기대효과, 제약, 근거 evidence_id, 비교 조건(실험 또는 적용 조건)을 표시하세요.
직접 비교가 어려우면 조건 차이 자체를 평가 결과로 남기고, 근거가 없으면 공개 정보 부족이라고 표시하세요.
마지막에 경쟁 관계와 보완 가능성을 구분하세요. JSON으로 반환하세요.

근거:
{compact_evidence(evidence)}
"""
        analysis = ask_json("당신은 데이터센터 LLM 서빙 시스템 전문가입니다.", prompt)
        print(f"[4/6] 도메인 평가 완료 - 근거 {len(evidence)}건")
        return {"domain_analysis": analysis, "references": evidence}
    except Exception as error:
        return {"domain_analysis": {}, "errors": [f"{agent_name}: {error}"]}
