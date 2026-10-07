"""시장 평가 Agent: 수요·채택·생태계·경제성. (담당: 시장 평가)"""
from __future__ import annotations

from config import FAST_MODE, WEB_MAX_RESULTS
from evidence import compact_evidence
from llm import ask_json
from search import web_search
from state import AgentState


def market_evaluation_agent(state: AgentState) -> dict:
    """검색 근거를 수집해 소프트웨어·하드웨어 시장 분석을 반환한다.

    ``state``는 다른 Agent와 동일한 호출 규약을 유지하기 위해 전달받는다.
    현재 시장 평가 단계에서는 상태값을 직접 사용하지 않는다.
    """
    agent_name = "market_evaluation"
    print("[2/6] 시장 평가 Agent 시작")
    try:
        # 기술별 검색 결과를 한 목록에 모아 LLM 분석과 최종 참고 자료에 함께 사용한다.
        evidence = []
        queries_by_technology = {
            "DeepSeek-V2 MLA": {
                "official": [
                    "DeepSeek-V2 MLA official deployment documentation",
                    "site:github.com DeepSeek-V2 MLA serving",
                ],
                "independent": [
                    "DeepSeek-V2 MLA independent benchmark deployment",
                    "DeepSeek-V2 MLA inference evaluation research",
                ],
            },
            "ITME": {
                "official": [
                    "SK hynix ITME CXL official",
                    "CXL Consortium LLM memory expansion",
                ],
                "independent": [
                    "ITME CXL memory independent evaluation",
                    "CXL KV cache deployment independent analysis",
                ],
            },
        }

        for technology, query_groups in queries_by_technology.items():
            # FAST_MODE에서도 공식/독립 유형을 각각 한 번은 검색한다.
            for technology_queries in query_groups.values():
                if FAST_MODE:
                    technology_queries = technology_queries[:1]
                for query in technology_queries:
                    search_results = web_search(
                        query,
                        agent_name,
                        technology,
                        max_results=WEB_MAX_RESULTS,
                    )
                    evidence.extend(search_results)

        # 분석 범위와 금지 사항을 명시해 과장되거나 근거 없는 시장 추정을 방지한다.
        prompt = f"""
검색 근거를 바탕으로 두 기술의 시장 수요, 상용화·채택, 생태계 지원, 도입 장벽, 경제성(공개된 비용 근거와 추가 투자 요구)을 분석하세요.
기업 홍보 주장과 독립적인 검증 결과를 구분하고 근거가 없는 내용은 공개 정보 부족이라고 표시하세요.
전체 AI 시장 규모를 개별 기술의 시장 규모처럼 사용하지 마세요. 공개된 비용 자료가 없으면 임의의 절감률을 만들지 마세요.
출처 등급을 구분하여 분석하세요.
- 핵심 성능 수치와 기술 구조는 1등급 원 논문을 우선하세요.
- 제품 지원과 공식 공개 사실은 2등급 공식 자료를 사용하세요.
- 시장 채택과 운영 효과는 3등급 독립 출처로 교차 확인하세요.
- 4~5등급 자료는 보조 설명으로만 사용하세요.
- 기업 공식 자료만 있는 주장은 '기업 공식 발표 기준'이라고 표시하세요.
- 독립 출처로 교차 확인되지 않은 시장 전망과 비용 절감 효과를 확정하지 마세요.
- 서로 다른 도메인의 출처가 2개 미만이면 '교차 검증 부족'으로 표시하세요.
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
        # 모델이 최상위 래퍼를 포함하거나 생략하는 두 응답 형태를 모두 허용한다.
        analysis = analysis.get("market_analysis", analysis)
        print(f"[2/6] 시장 평가 완료 - 근거 {len(evidence)}건")
        return {"market_analysis": analysis, "references": evidence}
    except Exception as error:
        # 파이프라인 전체가 중단되지 않도록 Agent 단위의 오류 정보로 반환한다.
        return {"market_analysis": {}, "errors": [f"{agent_name}: {error}"]}
