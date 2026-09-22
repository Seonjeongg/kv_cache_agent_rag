"""이해관계자 평가 Agent: 관점별 기대효과·우려. (담당: 이해관계자)"""
from __future__ import annotations

from config import FAST_MODE, WEB_MAX_RESULTS
from evidence import compact_evidence
from llm import ask_json
from search import web_search
from state import AgentState

STAKEHOLDER_QUERY_TEMPLATES = [
    "{technology} developer implementation limitations",
    "{technology} cloud provider deployment cost",
    "{technology} hardware vendor ecosystem response",
]


def stakeholder_evaluation_agent(state: AgentState) -> dict:
    agent_name = "stakeholder_evaluation"
    print("[3/6] 이해관계자 평가 Agent 시작")
    try:
        evidence = []
        for technology in ["DeepSeek-V2 MLA", "ITME"]:
            queries = [template.format(technology=technology) for template in STAKEHOLDER_QUERY_TEMPLATES]
            if FAST_MODE:
                queries = queries[:1]
            for query in queries:
                evidence.extend(web_search(query, agent_name, technology, max_results=WEB_MAX_RESULTS))

        # 근거 없이 그럴듯한 이해관계자 반응을 지어내지 않도록, 없으면 고정 문구를 쓰게 강제함
        prompt = f"""
다음 근거를 기술별·이해관계자별로 구분하세요. software는 클라우드 사업자, LLM 개발자, 서비스 개발자를,
hardware는 클라우드 사업자, 하드웨어 업체, 도입 기업·운영자를 평가하세요.
각 이해관계자의 기대효과, 우려, 도입 장벽을 분리하고 evidence_id를 연결하세요.
직접적인 발언이나 도입 사례를 찾지 못하면 "직접 반응 근거 없음"이라고 표시하세요.
자료에서 직접 확인되지 않는 이해관계자의 반응을 만들어내지 마세요. JSON으로 반환하세요.

반환 JSON 구조:
{{
  "stakeholder_analysis": {{
        "software": {{
            "cloud_provider": {{"expectation": "", "concern": "", "evidence_ids": []}},
            "llm_developer": {{"expectation": "", "concern": "", "evidence_ids": []}},
            "service_developer": {{"expectation": "", "concern": "", "evidence_ids": []}}
        }},
        "hardware": {{
            "cloud_provider": {{"expectation": "", "concern": "", "evidence_ids": []}},
            "hardware_vendor": {{"expectation": "", "concern": "", "evidence_ids": []}},
            "adopter_operator": {{"expectation": "", "concern": "", "evidence_ids": []}}
        }}
  }}
}}

근거:
{compact_evidence(evidence)}
"""
        analysis = ask_json("당신은 중립적인 기술 이해관계자 분석가입니다.", prompt)
        analysis = analysis.get("stakeholder_analysis", analysis)
        print(f"[3/6] 이해관계자 평가 완료 - 근거 {len(evidence)}건")
        return {"stakeholder_analysis": analysis, "references": evidence}
    except Exception as error:
        return {"stakeholder_analysis": {}, "errors": [f"{agent_name}: {error}"]}
