"""종합 평가, 근거 검증 Judge, 보고서 생성. (담당: 종합·검증·보고서)"""
from __future__ import annotations

import json
import re

from agents.technical_research import TERM_GLOSSARY
from config import FAST_MODE, MAX_RETRIES, REPORT_NUM_PREDICT
from evidence import collect_evidence_ids, format_references
from llm import ask_json
from state import AgentState

VALID_EVIDENCE_ID = re.compile(r"^(?:rag|web)-[0-9a-f]{12}$")


def synthesis_agent(state: AgentState) -> dict:
    print("[5/6] 종합 평가 Agent 시작")
    payload = {
        "technical": state.get("technical_analysis", {}),
        "trl": state.get("trl_analysis", {}),
        "market": state.get("market_analysis", {}),
        "stakeholder": state.get("stakeholder_analysis", {}),
        "domain": state.get("domain_analysis", {}),
    }
    prompt = f"""
아래 Agent 결과만 사용해 관점별 일치점, 상충점, 보완 가능성, 공개 정보의 한계를 종합하세요.
승자나 추천 기술을 결정하지 말고 새로운 사실을 추가하지 마세요.
각 결론에 근거가 된 evidence_id를 유지하세요. JSON으로 반환하세요.

Agent 결과:
{json.dumps(payload, ensure_ascii=False)[:30000]}
"""
    synthesis = ask_json("당신은 중립적인 기술 평가 종합 Agent입니다.", prompt)
    print("[5/6] 종합 평가 완료")
    return {"synthesis": synthesis}


def validation_judge(state: AgentState) -> dict:
    print("[Judge] 근거와 누락 검사 시작")
    missing = []
    retry_targets = []
    required = {
        "technical_analysis": "technical_research",
        "trl_analysis": "technical_research",
        "market_analysis": "market_evaluation",
        "stakeholder_analysis": "stakeholder_evaluation",
        "domain_analysis": "domain_evaluation",
        "synthesis": "synthesis",
    }

    for key, target in required.items():
        if not state.get(key):
            missing.append(f"{key} 누락")
            retry_targets.append(target)

    available_ids = {item.get("evidence_id") for item in state.get("references", [])}
    used_ids = set()
    for key in required:
        used_ids.update(collect_evidence_ids(state.get(key, {})))
    malformed_ids = sorted(item for item in used_ids if not VALID_EVIDENCE_ID.fullmatch(item))
    unknown_ids = sorted(
        item for item in used_ids
        if VALID_EVIDENCE_ID.fullmatch(item) and item not in available_ids
    )
    if malformed_ids:
        missing.append(f"형식이 잘못된 evidence_id: {malformed_ids[:10]}")
    if unknown_ids:
        missing.append(f"존재하지 않는 evidence_id: {unknown_ids[:10]}")

    if not state.get("references"):
        missing.append("Reference 근거 누락")

    retry_count = state.get("retry_count", 0)
    # 모델이 만든 형식 오류는 재조사로 해결되지 않으므로 한계로 기록하고 종료합니다.
    retryable_missing = bool(retry_targets or unknown_ids or not state.get("references"))
    if missing and retryable_missing and retry_count < MAX_RETRIES:
        status = "retry"
        retry_count += 1
    elif missing:
        status = "pass_with_limitations"
    else:
        status = "pass"

    print(f"[Judge] 결과={status}, 누락={len(missing)}, retry={retry_count}")
    for reason in missing:
        print(f"  - {reason}")
    return {
        "validation_result": status,
        "missing_evidence": missing,
        "retry_targets": sorted(set(retry_targets)),
        "retry_count": retry_count,
    }


def report_generation_agent(state: AgentState) -> dict:
    print("[6/6] 보고서 생성 Agent 시작")
    payload = {
        "selection_reason": state["selection_reason"],
        "technical": state.get("technical_analysis", {}),
        "trl": state.get("trl_analysis", {}),
        "market": state.get("market_analysis", {}),
        "stakeholder": state.get("stakeholder_analysis", {}),
        "domain": state.get("domain_analysis", {}),
        "synthesis": state.get("synthesis", {}),
        "validation": {
            "result": state.get("validation_result"),
            "limitations": state.get("missing_evidence", []),
        },
    }
    payload_limit = 12000 if FAST_MODE else 36000
    prompt = f"""
당신은 연구자와 데이터센터 의사결정자를 위한 중립적 기술평가 보고서를 작성합니다.
아래의 검증된 Agent 분석 결과만 사용해, 배경과 근거에서 판단과 시사점으로 이어지는 보고서 본문을 작성하세요.
작성 과정이나 생각은 출력하지 말고 지정된 JSON 객체만 반환하세요.

[공통 작성 원칙]
- JSON 키를 제외한 모든 문자열 값은 반드시 한국어로 작성하세요.
- 영어 근거도 한국어로 해석하되, 기술명·고유명사·약어·논문 제목은 원문 표기를 허용합니다.
- 하나의 문단에서 사실, 해석, 판단을 섞지 말고 각각 구분해 서술하세요.
- 모든 수치·성능·채택·비용·TRL 주장은 문장 끝에 해당 evidence_id를 [rag-...] 또는 [web-...] 형식으로 붙이세요.
- 시장성 절과 이해관계자 절은 반드시 수집된 웹 근거의 [web-...] ID를 포함하세요. 웹 근거가 없을 때만 공개 정보 부족이라고 쓰세요.
- 입력에 없는 수치, 기업 도입 사례, 시장 반응, 운영 결과를 추론해 사실처럼 쓰지 마세요.
- 직접 근거가 없으면 '공개 정보 부족'이라고 쓰고, 무엇이 부족한지와 판단에 미치는 영향을 설명하세요.
- 두 기술의 실험 환경이 다르면 수치를 직접 우열 비교하지 말고 비교 조건의 차이를 먼저 설명하세요.
- 특정 기술을 추천하거나 승자를 정하지 말고, 적용 조건에 따른 장점·제약·보완 가능성을 균형 있게 작성하세요.

[보고서 구성과 분량]
- SUMMARY는 핵심 결론, 중요한 차이, 공통 한계, 추가 검증 과제를 포함한 8~10문장으로 작성하세요.
- SUMMARY를 제외한 모든 항목은 제목 없이 자연스러운 보고서 본문으로 작성하세요.
- 각 항목은 최소 2개 문단, 8~12문장, 700자 이상을 목표로 하세요. 단순한 문장 반복으로 분량을 채우지 마세요.
- 각 문단은 '주장 또는 관찰 → 근거 → 해석 → 판단의 한계' 순서로 전개하세요.

[절별 작성 지시]
- background: 데이터센터 LLM 서빙에서 KV cache가 문제가 되는 이유, 메모리·긴 컨텍스트·동시 사용자와의 관계, 본 보고서의 비교 범위와 평가 관점을 설명하세요.
- selection: 소프트웨어 계층의 MLA와 하드웨어·메모리 계층의 ITME를 선택한 이유, 동일 병목에 대한 접근 차이, 비교 가능한 항목과 직접 비교가 어려운 항목을 설명하세요.
- deepseek_overview: MLA의 latent compression, decoupled RoPE 구성, KV cache 저장량 변화, 모델 적용 범위, 성능 근거, 구현·서빙 요구사항과 제약을 설명하세요.
- itme_overview: ITME의 HBM-CXL hybrid memory 구조, KV cache와 모델 데이터의 배치 또는 이동 방식, 지연·대역폭 고려사항, 적용 범위, 인프라 요구사항과 제약을 설명하세요.
- trl: 각 기술에 대해 확인된 실험 환경, 프로토타입·코드·시스템 통합·실제 운영 근거를 나누어 TRL을 추정하세요. 추정할 수 없는 단계는 이유를 명시하세요.
- market: 잠재 수요가 발생하는 비용·메모리 문제, 공개된 채택·생태계 자료, 구매·인프라 투자 요인, 도입 장벽을 기술별로 구분하세요.
- stakeholder: 클라우드 사업자, LLM 개발자, 서비스 개발자, 하드웨어 업체, 운영자의 기대효과·우려·도입 요구사항을 각각 비교하세요.
- domain: HBM 사용량, 긴 컨텍스트, 동시 사용자, TTFT, 처리량, 정확도 영향, 데이터 이동 지연, 호환성, 운영 복잡도·비용을 기준별로 비교하고 조건 차이를 기록하세요.
- comparison_conflicts: 두 기술이 경쟁하는 지점과 함께 사용할 수 있는 지점을 분리하세요. 직접 우열 판단이 왜 위험한지 실험 단위와 시스템 계층 차이로 설명하세요.
- implications: 기술 선택을 대신하지 말고, 어떤 서빙 조건에서 어떤 검증 질문을 우선해야 하는지와 PoC·벤치마크 후속 과제를 제시하세요.
- limitations: 논문·웹 자료의 범위, 검색 품질, 실험 조건 불일치, 공개되지 않은 비용·운영·상용화 정보, LLM 생성 해석의 한계를 구체적으로 정리하세요.

{TERM_GLOSSARY}

반환 JSON 구조:
{{
  "summary": "",
  "background": "",
  "selection": "",
  "deepseek_overview": "",
  "itme_overview": "",
  "trl": "",
  "market": "",
  "stakeholder": "",
  "domain": "",
  "comparison_conflicts": "",
  "implications": "",
  "limitations": ""
}}

분석 결과:
{json.dumps(payload, ensure_ascii=False)[:payload_limit]}
"""
    report_data = ask_json(
        "당신은 근거 기반의 중립적인 기술 평가 보고서 작성자입니다.",
        prompt,
        num_predict=REPORT_NUM_PREDICT,
    )

    if report_data.get("parse_error"):
        report_data = {
            "summary": "보고서 구조화 생성에 실패했습니다.",
            "background": "공개 정보 부족",
            "selection": state["selection_reason"],
            "deepseek_overview": "공개 정보 부족",
            "itme_overview": "공개 정보 부족",
            "trl": "공개 정보 부족",
            "market": "공개 정보 부족",
            "stakeholder": "공개 정보 부족",
            "domain": "공개 정보 부족",
            "comparison_conflicts": "공개 정보 부족",
            "implications": "공개 정보 부족",
            "limitations": "LLM의 JSON 출력 형식을 해석하지 못했습니다.",
        }

    report_sections = [
        "background", "selection", "deepseek_overview", "itme_overview",
        "trl", "market", "stakeholder", "domain", "comparison_conflicts",
        "implications", "limitations",
    ]
    short_sections = {
        name: report_data.get(name, "")
        for name in report_sections
        if len(str(report_data.get(name, ""))) < 650
    }
    if short_sections:
        expansion_prompt = f"""
    다음 보고서 절 초안은 기술평가 보고서로서 설명과 근거가 부족합니다.
    입력된 분석 결과와 기존 evidence_id만 사용해 각 절을 연구자·데이터센터 의사결정자가 읽을 수 있는 본문으로 확장하세요.
    모든 문자열은 한국어로 작성하고, JSON 키는 입력 키를 그대로 유지하세요.
    각 절은 최소 700자, 2개 이상의 문단으로 작성하세요.
    각 문단은 관찰 또는 주장, 근거, 해석, 판단의 한계 순서로 전개하세요.
    새로운 수치·사례·기업 반응을 만들지 말고, 주장 뒤에는 기존 evidence_id를 유지하세요.
    근거가 부족하면 단순히 문장을 반복하지 말고, 확인되지 않은 정보와 그로 인한 비교·판단의 한계를 설명하세요.

확장할 절 초안:
{json.dumps(short_sections, ensure_ascii=False)}

참고할 분석 결과:
{json.dumps(payload, ensure_ascii=False)[:30000]}
"""
        expanded = ask_json(
            "당신은 한국어 기술 평가 보고서의 부족한 절을 보완하는 편집자입니다.",
            expansion_prompt,
            num_predict=5000,
        )
        for name in short_sections:
            value = expanded.get(name)
            if isinstance(value, str) and len(value) > len(str(report_data.get(name, ""))):
                report_data[name] = value

    def section(name: str) -> str:
        value = report_data.get(name) or "공개 정보 부족"
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)

    body = f"""# SUMMARY

{section('summary')}

# 1. 분석 배경

{section('background')}

# 2. 기술 선정

{section('selection')}

# 3. 기술 개요

## 3.1 DeepSeek-V2 MLA

{section('deepseek_overview')}

## 3.2 ITME

{section('itme_overview')}

# 4. 관점별 평가

## 4.1 기술 성숙도 및 TRL

{section('trl')}

## 4.2 시장성

{section('market')}

## 4.3 이해관계자

{section('stakeholder')}

## 4.4 데이터센터·클라우드 적용성

{section('domain')}

# 5. 관점별 비교 및 상충 지점

{section('comparison_conflicts')}

# 6. 시사점

{section('implications')}

# 7. 분석의 한계

{section('limitations')}"""
    used_ids = set(re.findall(r"(?:rag|web)-[a-f0-9]{12}", body))
    # 시장성·이해관계자 Agent가 사용한 웹 근거는 본문에서 ID가 누락되어도
    # 최종 참고문헌에서 빠지지 않도록 분석 결과의 evidence_ids를 함께 반영합니다.
    used_ids.update(collect_evidence_ids(state.get("market_analysis", {})))
    used_ids.update(collect_evidence_ids(state.get("stakeholder_analysis", {})))
    references = format_references(state.get("references", []), used_ids=used_ids)
    report = body + "\n\n# REFERENCE\n\n" + references
    print("[6/6] 보고서 생성 완료")
    return {"report": report}
