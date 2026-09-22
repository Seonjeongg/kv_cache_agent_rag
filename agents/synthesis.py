"""종합 평가, 근거 검증 Judge, 보고서 생성 Agent."""
from __future__ import annotations

import json
import re
from typing import Any

from agents.technical_research import TERM_GLOSSARY
from config import FAST_MODE, MAX_RETRIES, REPORT_NUM_PREDICT
from evidence import collect_evidence_ids, format_references
from llm import ask_json
from state import AgentState


VALID_EVIDENCE_ID = re.compile(r"^(?:rag|web)-[0-9a-f]{12}$")
REPORT_EVIDENCE_ID = re.compile(r"(?:rag|web)-[0-9a-f]{12}")

REQUIRED_ANALYSES = {
    "technical_analysis": "technical_research",
    "trl_analysis": "technical_research",
    "market_analysis": "market_evaluation",
    "stakeholder_analysis": "stakeholder_evaluation",
    "domain_analysis": "domain_evaluation",
    "synthesis": "synthesis",
}

ANALYSIS_PAYLOAD_KEYS = {
    "technical": "technical_analysis",
    "trl": "trl_analysis",
    "market": "market_analysis",
    "stakeholder": "stakeholder_analysis",
    "domain": "domain_analysis",
}

REPORT_SECTION_NAMES = (
    "background",
    "selection",
    "deepseek_overview",
    "itme_overview",
    "trl",
    "market",
    "stakeholder",
    "domain",
    "comparison_conflicts",
    "implications",
    "limitations",
)

MIN_REPORT_SECTION_LENGTH = 650
EXPANSION_NUM_PREDICT = max(REPORT_NUM_PREDICT, 5000)


def _json(value: Any, limit: int | None = None) -> str:
    """JSON 직렬화와 프롬프트 길이 제한을 한곳에서 처리합니다."""
    serialized = json.dumps(value, ensure_ascii=False)
    return serialized[:limit] if limit else serialized


def _unwrap_analysis(state_key: str, value: Any) -> Any:
    """Agent가 동일 키를 한 번 더 감싸 반환한 결과를 정상화합니다.

    예: {"market_analysis": {"software": ...}} -> {"software": ...}
    """
    while (
        isinstance(value, dict)
        and set(value) == {state_key}
        and isinstance(value[state_key], dict)
    ):
        value = value[state_key]
    return value


def _state_analysis(state: AgentState, state_key: str) -> dict:
    value = _unwrap_analysis(state_key, state.get(state_key, {}))
    return value if isinstance(value, dict) else {}


def _analysis_payload(state: AgentState) -> dict:
    return {
        payload_key: _state_analysis(state, state_key)
        for payload_key, state_key in ANALYSIS_PAYLOAD_KEYS.items()
    }


def has_usable_analysis(value: Any) -> bool:
    """파싱 오류, 빈 객체, 빈 하위 분석을 정상 결과로 처리하지 않습니다."""
    if not isinstance(value, dict) or not value or value.get("parse_error"):
        return False

    meaningful_leaf_exists = False
    for item in value.values():
        if isinstance(item, dict):
            if not has_usable_analysis(item):
                return False
            meaningful_leaf_exists = True
        elif isinstance(item, list):
            meaningful_leaf_exists = meaningful_leaf_exists or bool(item)
        elif item not in ("", None):
            meaningful_leaf_exists = True

    return meaningful_leaf_exists


def synthesis_agent(state: AgentState) -> dict:
    print("[5/6] 종합 평가 Agent 시작")
    prompt = f"""
아래 Agent 결과만 사용해 관점별 일치점, 상충점, 보완 가능성, 공개 정보의 한계를 종합하세요.
승자나 추천 기술을 결정하지 말고 새로운 사실을 추가하지 마세요.
각 결론에 근거가 된 evidence_id를 유지하세요. JSON으로 반환하세요.

Agent 결과:
{_json(_analysis_payload(state), limit=30000)}
"""
    synthesis = ask_json(
        "당신은 중립적인 기술 평가 종합 Agent입니다.",
        prompt,
    )
    print("[5/6] 종합 평가 완료")
    return {"synthesis": synthesis}


def _available_evidence_ids(state: AgentState) -> set[str]:
    return {
        evidence_id
        for item in state.get("references", [])
        if (evidence_id := item.get("evidence_id"))
    }


def _used_evidence_ids(state: AgentState) -> set[str]:
    used_ids: set[str] = set()
    for state_key in REQUIRED_ANALYSES:
        analysis = _state_analysis(state, state_key)
        used_ids.update(collect_evidence_ids(analysis))
    return used_ids


def _validate_evidence_ids(state: AgentState) -> tuple[list[str], list[str]]:
    available_ids = _available_evidence_ids(state)
    used_ids = _used_evidence_ids(state)

    malformed_ids = sorted(
        evidence_id
        for evidence_id in used_ids
        if not VALID_EVIDENCE_ID.fullmatch(evidence_id)
    )
    unknown_ids = sorted(
        evidence_id
        for evidence_id in used_ids
        if VALID_EVIDENCE_ID.fullmatch(evidence_id)
        and evidence_id not in available_ids
    )
    return malformed_ids, unknown_ids


def validation_judge(state: AgentState) -> dict:
    print("[Judge] 근거와 누락 검사 시작")
    missing: list[str] = []
    retry_targets: list[str] = []

    for state_key, target in REQUIRED_ANALYSES.items():
        analysis = _state_analysis(state, state_key)
        if not has_usable_analysis(analysis):
            missing.append(f"{state_key} 누락 또는 오류")
            retry_targets.append(target)

    malformed_ids, unknown_ids = _validate_evidence_ids(state)
    if malformed_ids:
        missing.append(f"형식이 잘못된 evidence_id: {malformed_ids[:10]}")
    if unknown_ids:
        missing.append(f"존재하지 않는 evidence_id: {unknown_ids[:10]}")

    references_missing = not state.get("references")
    if references_missing:
        missing.append("Reference 근거 누락")

    retry_count = state.get("retry_count", 0)
    retryable = bool(retry_targets or unknown_ids or references_missing)

    if missing and retryable and retry_count < MAX_RETRIES:
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


def _report_payload(state: AgentState) -> dict:
    payload = {
        "selection_reason": state.get("selection_reason", "공개 정보 부족"),
        **_analysis_payload(state),
        "synthesis": _state_analysis(state, "synthesis"),
        "validation": {
            "result": state.get("validation_result"),
            "limitations": state.get("missing_evidence", []),
        },
    }
    return payload


def _report_prompt(payload: dict) -> str:
    payload_limit = 12000 if FAST_MODE else 36000
    return f"""
당신은 연구자와 데이터센터 의사결정자를 위한 중립적 기술평가 보고서를 작성합니다.
아래의 검증된 Agent 분석 결과만 사용해 배경과 근거에서 판단과 시사점으로 이어지는 보고서 본문을 작성하세요.
작성 과정이나 생각은 출력하지 말고 지정된 JSON 객체만 반환하세요.

[공통 작성 원칙]
- JSON 키를 제외한 모든 문자열 값은 반드시 한국어로 작성하세요.
- 기술명·고유명사·약어·논문 제목은 원문 표기를 허용합니다.
- 사실, 해석, 판단을 구분해 서술하세요.
- 모든 수치·성능·채택·비용·TRL 주장 뒤에 evidence_id를 붙이세요.
- 시장성·이해관계자 절에는 수집된 웹 근거의 [web-...] ID를 포함하세요.
- 입력에 없는 수치, 도입 사례, 시장 반응, 운영 결과를 만들지 마세요.
- 직접 근거가 없으면 '공개 정보 부족'과 판단에 미치는 영향을 설명하세요.
- 실험 환경이 다르면 수치를 직접 우열 비교하지 마세요.
- 특정 기술을 추천하지 말고 적용 조건별 장점·제약·보완 가능성을 균형 있게 작성하세요.

[분량]
- SUMMARY는 핵심 결론, 차이, 공통 한계, 추가 검증 과제를 포함한 8~10문장으로 작성하세요.
- 나머지 항목은 제목 없이 최소 2개 문단, 8~12문장, 700자 이상을 목표로 작성하세요.
- 문장 반복으로 분량을 채우지 마세요.

[절별 작성 지시]
- background: KV cache 병목과 메모리·긴 컨텍스트·동시 사용자의 관계, 비교 범위를 설명하세요.
- selection: MLA와 ITME의 선정 이유, 접근 계층 차이, 비교 가능·불가능 항목을 설명하세요.
- deepseek_overview: latent compression, decoupled RoPE, 저장량 변화, 성능 근거와 제약을 설명하세요.
- itme_overview: HBM-CXL hybrid memory, 데이터 배치·이동, 지연·대역폭, 인프라 제약을 설명하세요.
- trl: 실험 환경, 프로토타입·코드·시스템 통합·실제 운영 근거를 구분해 TRL을 추정하세요.
- market: 수요, 채택, 생태계, 투자 요인과 도입 장벽을 기술별로 구분하세요.
- stakeholder: 이해관계자별 기대효과, 우려와 도입 요구사항을 비교하세요.
- domain: HBM, 긴 컨텍스트, 동시 사용자, TTFT, 처리량, 정확도, 지연, 호환성, 비용을 비교하세요.
- comparison_conflicts: 경쟁·보완 지점을 분리하고 직접 우열 판단이 위험한 이유를 설명하세요.
- implications: 조건별 검증 질문과 PoC·벤치마크 후속 과제를 제시하세요.
- limitations: 자료 범위, 검색 품질, 조건 불일치, 비공개 정보와 LLM 해석 한계를 정리하세요.

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
{_json(payload, limit=payload_limit)}
"""


def _fallback_report_data(state: AgentState) -> dict[str, str]:
    fallback = {name: "공개 정보 부족" for name in REPORT_SECTION_NAMES}
    fallback.update({
        "summary": "보고서 구조화 생성에 실패했습니다.",
        "selection": state.get("selection_reason", "공개 정보 부족"),
        "limitations": "LLM의 JSON 출력 형식을 해석하지 못했습니다.",
    })
    return fallback


def _short_sections(report_data: dict) -> dict[str, str]:
    return {
        name: str(report_data.get(name, ""))
        for name in REPORT_SECTION_NAMES
        if len(str(report_data.get(name, ""))) < MIN_REPORT_SECTION_LENGTH
    }


def _expand_short_sections(
    report_data: dict,
    payload: dict,
) -> dict:
    short_sections = _short_sections(report_data)
    if not short_sections:
        return report_data

    expansion_prompt = f"""
다음 보고서 절 초안을 입력된 분석 결과와 기존 evidence_id만 사용해 보완하세요.
모든 문자열은 한국어로 작성하고 JSON 키는 그대로 유지하세요.
각 절은 최소 700자, 2개 이상의 문단으로 작성하세요.
새로운 수치·사례·기업 반응을 만들지 말고 기존 evidence_id를 유지하세요.
근거가 부족하면 확인되지 않은 정보와 그것이 판단에 미치는 영향을 설명하세요.

확장할 절 초안:
{_json(short_sections)}

참고할 분석 결과:
{_json(payload, limit=30000)}
"""
    expanded = ask_json(
        "당신은 한국어 기술 평가 보고서의 부족한 절을 보완하는 편집자입니다.",
        expansion_prompt,
        num_predict=EXPANSION_NUM_PREDICT,
    )
    if expanded.get("parse_error"):
        return report_data

    for name, original in short_sections.items():
        candidate = expanded.get(name)
        if isinstance(candidate, str) and len(candidate) > len(original):
            report_data[name] = candidate
    return report_data


def _section(report_data: dict, name: str) -> str:
    value = report_data.get(name) or "공개 정보 부족"
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)


def _render_report_body(report_data: dict) -> str:
    section = lambda name: _section(report_data, name)
    return f"""# SUMMARY

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


def _reference_ids(body: str, state: AgentState) -> set[str]:
    used_ids = set(REPORT_EVIDENCE_ID.findall(body))

    # 시장·이해관계자 분석에 실제 연결된 웹 근거가 본문 생성 과정에서
    # 누락되더라도 참고문헌에서 사라지지 않도록 포함합니다.
    for state_key in ("market_analysis", "stakeholder_analysis"):
        used_ids.update(collect_evidence_ids(_state_analysis(state, state_key)))
    return used_ids


def report_generation_agent(state: AgentState) -> dict:
    print("[6/6] 보고서 생성 Agent 시작")
    payload = _report_payload(state)
    report_data = ask_json(
        "당신은 근거 기반의 중립적인 기술 평가 보고서 작성자입니다.",
        _report_prompt(payload),
        num_predict=REPORT_NUM_PREDICT,
    )

    if report_data.get("parse_error"):
        report_data = _fallback_report_data(state)
    report_data = _expand_short_sections(report_data, payload)

    body = _render_report_body(report_data)
    references = format_references(
        state.get("references", []),
        used_ids=_reference_ids(body, state),
    )
    report = f"{body}\n\n# REFERENCE\n\n{references}"
    print("[6/6] 보고서 생성 완료")
    return {"report": report}
