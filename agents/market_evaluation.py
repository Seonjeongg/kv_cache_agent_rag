"""시장 평가 Agent: 수요·채택·생태계·경제성. (담당: 시장 평가)"""
from __future__ import annotations

from config import FAST_MODE, WEB_MAX_RESULTS
from evidence import compact_evidence
from llm import ask_json
from search import web_search
from state import AgentState


def market_evaluation_agent(state: AgentState) -> dict:
    agent_name = "market_evaluation"
    print("[2/6] 시장 평가 Agent 시작")
    try:
        evidence = []
        queries = {
            "DeepSeek-V2 MLA": [
                "DeepSeek-V2 MLA deployment framework adoption official",
                "MLA inference serving ecosystem official",
                "DeepSeek-V2 MLA cost savings GPU infrastructure investment",
            ],
            "ITME": [
                "ITME CXL hybrid memory LLM inference adoption",
                "CXL memory expansion LLM serving ecosystem official",
                "ITME CXL memory cost infrastructure investment",
            ],
        }
        for technology, technology_queries in queries.items():
            if FAST_MODE:
                technology_queries = technology_queries[:1]
            for query in technology_queries:
                evidence.extend(web_search(query, agent_name, technology, max_results=WEB_MAX_RESULTS))

        prompt = f"""
검색 근거를 바탕으로 두 기술의 시장 수요, 상용화·채택, 생태계 지원, 도입 장벽, 경제성(공개된 비용 근거와 추가 투자 요구)을 분석하세요.
기업 홍보 주장과 독립적인 검증 결과를 구분하고 근거가 없는 내용은 공개 정보 부족이라고 표시하세요.
전체 AI 시장 규모를 개별 기술의 시장 규모처럼 사용하지 마세요. 공개된 비용 자료가 없으면 임의의 절감률을 만들지 마세요.
모든 항목에 evidence_id 목록을 포함하세요. JSON으로 반환하세요.

반환 JSON 구조:
{{
  "market_analysis": {{
    "software": {{"demand": "", "adoption": "", "ecosystem": "", "adoption_barrier": "", "economics": "", "evidence_ids": []}},
    "hardware": {{"demand": "", "adoption": "", "ecosystem": "", "adoption_barrier": "", "economics": "", "evidence_ids": []}}
  }}
}}

근거:
{compact_evidence(evidence)}
"""
        analysis = ask_json("당신은 중립적인 AI 인프라 시장 분석가입니다.", prompt)
        print(f"[2/6] 시장 평가 완료 - 근거 {len(evidence)}건")
        return {"market_analysis": analysis, "references": evidence}
    except Exception as error:
        return {"market_analysis": {}, "errors": [f"{agent_name}: {error}"]}
