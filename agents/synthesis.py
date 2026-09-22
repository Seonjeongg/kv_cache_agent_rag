"""종합 평가, 근거 검증 Judge, 보고서 생성. (담당: 종합·검증·보고서)"""
from __future__ import annotations

import json
import re

from agents.technical_research import TERM_GLOSSARY
from config import FAST_MODE, MAX_RETRIES, REPORT_NUM_PREDICT
from evidence import collect_evidence_ids, format_references
from llm import ask_json
from state import AgentState


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
    invalid_ids = sorted(used_ids - available_ids)
    if invalid_ids:
        missing.append(f"존재하지 않는 evidence_id: {invalid_ids[:10]}")

    if not state.get("references"):
        missing.append("Reference 근거 누락")

    retry_count = state.get("retry_count", 0)
    if missing and retry_count < MAX_RETRIES:
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
다음 검증된 분석 결과를 사용해 보고서 각 절의 본문을 작성하세요.
작성 과정이나 생각을 출력하지 말고 지정된 JSON 객체만 반환하세요.
특정 기술을 추천하거나 우열을 판정하지 말고 관점별 상충 지점을 유지하세요.
주요 주장 뒤에는 [evidence_id]를 넣고, 근거가 없으면 '공개 정보 부족'이라고 쓰세요.
입력에 없는 수치, 도입 사례, 시장 반응을 만들지 마세요.
SUMMARY는 1/2페이지 이내의 결과 요약이어야 하며 개요 소개가 아닙니다.
JSON이 잘리지 않도록 각 항목은 핵심 3~5문장으로 간결하게 작성하세요.

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
    references = format_references(state.get("references", []), used_ids=used_ids)
    report = body + "\n\n# REFERENCE\n\n" + references
    print("[6/6] 보고서 생성 완료")
    return {"report": report}
