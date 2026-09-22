"""기술 조사 Agent: 원리·성능·한계, TRL 추정"""
from __future__ import annotations

from config import AGENT_RAG_TOP_K, FAST_MODE
from evidence import compact_evidence, rag_evidence
from llm import ask_json
from state import AgentState
from agents.technology_selection import TECHNOLOGY_SPECS

# 기술 성숙도 수준(Technology Readiness Level, TRL) 정의
TRL_LEVELS = """TRL 1: 기초 원리 관찰
TRL 2: 기술 개념과 적용 가능성 정립
TRL 3: 핵심 기능의 개념 검증
TRL 4: 실험실에서 구성요소 검증
TRL 5: 관련 환경에서 구성요소 검증
TRL 6: 관련 환경에서 시스템·프로토타입 시연
TRL 7: 실제 운용 환경에서 프로토타입 시연
TRL 8: 완성된 시스템의 검증
TRL 9: 실제 운용 실적 확인"""

# 기술 용어집 정의
TERM_GLOSSARY = """용어집 (다른 말로 풀어쓰지 말고 아래 정의를 그대로 사용하세요):
- MLA = Multi-head Latent Attention
- ITME = Inference Tiered Memory Expansion
- CXL = Compute Express Link
- RoPE = Rotary Position Embedding
- HBM = High Bandwidth Memory
- TRL = Technology Readiness Level"""

# 기술 조사 Agent 정의
def technical_research_agent(_state: AgentState) -> dict:
    agent_name = "technical_research"
    print("[1/6] 기술 조사 Agent 시작")
    try:
        questions = [ # 기술 조사 질문 정의
            "What problem does the technology solve, and what is its core mechanism in the paper?",
            "How does the technology reduce or manage the number of cached key-value elements per token?",
            "What evaluation environment, baselines, metrics, and performance results are reported?",
            "What architecture constraints, limitations, and deployment requirements are reported?",
            "What evidence indicates system-scale evaluation, prototype validation, or technology readiness?",
        ]
        if FAST_MODE: # FAST_MODE가 활성화된 경우 질문 단순화
            questions = [ # 기술 조사 질문 단순화 정의
                "Explain the core mechanism, cached key-value memory reduction per token, evaluation conditions, reported performance, limitations, deployment requirements, and readiness evidence."
            ]
        analyses = {}
        trl_analyses = {}
        evidence = []
        for side, spec in TECHNOLOGY_SPECS.items(): # 기술 조사 수행
            technology = spec["name"]
            technology_evidence = []
            for question in questions: # 기술 조사 질문에 대한 근거 수집
                technology_evidence.extend(rag_evidence(question, technology, agent_name, top_k=AGENT_RAG_TOP_K))
            evidence.extend(technology_evidence)

            # 기술 조사 프롬프트 정의
            prompt = f"""
다음은 {technology}에 대한 논문 근거입니다. 다른 기술의 사실이나 근거를 섞지 말고 {technology}만 분석하세요.
수치를 쓸 때 evidence_id를 반드시 연결하세요. 서로 다른 실험 환경의 수치를 직접 우열 비교하지 마세요.

TRL은 아래 9단계 기준에 따라 공개 근거 기반으로만 추정하며, 확정이 아님을 명시하세요.
논문 발표만으로 특정 단계를 자동 부여하지 말고, 코드 공개·프로토타입·실험 환경·시스템 통합·실제 운용 근거를 함께 확인하세요.
관련 기업이나 주변 기술의 성숙도를 해당 기술의 TRL로 대신하지 마세요. 근거가 부족하면 추정을 유보하거나 범위와 한계를 표시하세요.

{TRL_LEVELS}

{TERM_GLOSSARY}

반환 JSON 구조:
{{
    "technical": {{"principle": "", "scope": "", "performance_claims": [], "limitations": [], "evidence_ids": []}},
    "trl": {{"estimated_trl": null, "reason": "", "evidence_ids": [], "disclaimer": "공개 정보 기반 추정"}}
}}

근거:
{compact_evidence(technology_evidence)} 
"""
            result = ask_json("당신은 중립적인 LLM 시스템 기술 분석가입니다.", prompt)
            analyses[side] = result.get("technical", {})
            trl_analyses[side] = result.get("trl", {})
        print(f"[1/6] 기술 조사 완료 - 근거 {len(evidence)}건")
        return { # 기술 조사 결과 반환
            "technical_analysis": analyses,
            "trl_analysis": trl_analyses,
            "references": evidence,
        }
    except Exception as error: # 기술 조사 중 오류 발생 시 오류 메시지 반환
        return {"technical_analysis": {}, "trl_analysis": {}, "errors": [f"{agent_name}: {error}"]}
