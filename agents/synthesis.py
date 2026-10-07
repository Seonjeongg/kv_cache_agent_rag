"""종합 평가, 근거 검증 Judge, 보고서 생성. (담당: 종합·검증·보고서)"""
from __future__ import annotations

import json
import re

from agents.technical_research import TERM_GLOSSARY
from config import FAST_MODE, MAX_RETRIES, REPORT_NUM_PREDICT
from evidence import build_numbered_references, collect_evidence_ids
from llm import ask_json
from state import AgentState

VALID_EVIDENCE_ID = re.compile(r"^(?:rag|web)-[0-9a-f]{12}$")
EVIDENCE_TOKEN = re.compile(r"(?:rag|web)-[0-9a-f]{12}")


def exclude_unregistered_evidence(text: str, available_ids: set[str]) -> str:
    """등록되지 않은 인용 ID만 제거하고 문장 내용은 보존합니다."""
    paragraphs = re.split(r"\n{2,}", str(text or "").strip())
    safe = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?。！？])\s+|\n+", paragraph)
        cleaned_sentences = []
        for sentence in sentences:
            invalid_ids = set(EVIDENCE_TOKEN.findall(sentence)) - available_ids
            for invalid_id in invalid_ids:
                sentence = sentence.replace(f"[{invalid_id}]", "")
            cleaned = EVIDENCE_TOKEN.sub(
                lambda match: "" if match.group() in invalid_ids else match.group(),
                sentence,
            ).strip()
            if invalid_ids:
                cleaned += " (해당 문장은 등록된 출처와 연결할 수 없어 추가 검증이 필요하다.)"
            if cleaned:
                cleaned_sentences.append(cleaned)
        safe.append(" ".join(cleaned_sentences))
    return "\n\n".join(item for item in safe if item).strip()


def _depth_fallback(name: str) -> str:
    if name == "implications":
        return (
            "이 분석의 시사점은 두 기술 중 하나를 선택하는 결론이 아니라, 적용 조건에 따라 무엇을 먼저 검증해야 하는지를 정리하는 데 있다. "
            "MLA처럼 모델 내부의 KV 표현과 저장량을 바꾸는 접근은 모델 구조, 어텐션 구현, 긴 컨텍스트 길이, 동시 요청 수를 함께 확인해야 한다. "
            "ITME처럼 메모리 계층과 데이터 이동 경로를 바꾸는 접근은 HBM과 CXL 사이의 이동량, 접근 지연, 대역폭, 버퍼 관리, 장치 호환성을 별도로 확인해야 한다. "
            "따라서 공개 자료에서 확인된 원리나 프로토타입 결과를 실제 서비스의 처리량과 비용으로 바로 일반화해서는 안 된다. "
            "두 접근을 검토할 때에는 먼저 동일한 모델, 하드웨어, 소프트웨어 버전, 입력 길이, 동시성 조건을 고정한 기준선을 마련해야 한다. "
            "그 다음 KV cache 사용량, HBM 점유량, 첫 토큰 지연시간, 토큰 처리량, tail latency, 데이터 이동량, 정확도 변화를 같은 방식으로 측정해야 한다. "
            "결과가 확인되더라도 운영 복잡도와 장애 대응 방식, 기존 서빙 스택과의 호환성, 추가 장비 및 개발 비용을 함께 기록해야 한다. "
            "이런 절차를 거쳐야만 기술의 장점과 제약을 동일한 판단 단위에서 비교할 수 있다."
            "\n\n"
            "후속 PoC에서는 기술별로 확인 가능한 주장과 아직 공개 정보가 부족한 주장을 분리해 실험 계획에 반영해야 한다. "
            "예를 들어 짧은 입력과 긴 입력, 낮은 동시성과 높은 동시성, 캐시 적중과 캐시 교체가 많은 상황을 나누어 측정하면 특정 조건에만 나타나는 효과를 구분할 수 있다. "
            "또한 단일 평균값만 기록하지 말고 반복 측정의 분산과 최악 구간을 함께 기록해야 한다. "
            "모델 정확도나 응답 품질이 유지되는지도 같은 질의 집합으로 확인해야 하며, 메모리 절감이 실제 비용 절감으로 이어지는지는 인프라 가격과 운영 정책을 포함해 별도로 계산해야 한다. "
            "현재 자료만으로는 이 후속 결과를 확정할 수 없으므로, 본 보고서의 시사점은 의사결정의 방향이 아니라 검증 우선순위와 측정 항목을 제안하는 수준으로 해석해야 한다."
        )
    return (
        "본 분석의 한계는 근거가 없다는 사실과 기술이 존재하지 않는다는 판단을 구분해야 한다는 점에서 시작한다. "
        "현재 보고서는 제공된 논문과 검색된 공개 자료를 바탕으로 작성되었으며, 모든 구현 버전, 내부 운영 지표, 계약 조건, 실제 구매 비용과 상용 서비스의 세부 설정을 확인한 것은 아니다. "
        "논문마다 모델 크기, 하드웨어, 입력 길이, 동시성, 측정 방법이 다를 수 있으므로 서로 다른 실험의 수치를 직접 비교하면 해석이 왜곡될 수 있다. "
        "웹 자료는 게시 시점과 검색 결과의 품질에 영향을 받으며, 자료가 갱신되거나 접근이 제한될 가능성도 있다. "
        "또한 검색된 자료에 특정 관점의 정보가 더 많이 포함되면 시장성이나 이해관계자 분석이 실제보다 풍부하게 보일 수 있다. "
        "이 보고서의 문장은 입력된 근거를 요약하고 연결한 결과이므로, 원문이 말하지 않은 운영 효과나 도입 가능성을 사실처럼 확정하지 않아야 한다. "
        "등록되지 않은 근거 ID 표기는 제거하고 해당 문장에 추가 검증 필요성을 표시했으며, 이는 해당 주장이 거짓이라는 뜻이 아니라 현재 자료로 완전히 추적할 수 없다는 뜻이다."
        "\n\n"
        "따라서 최종 판단 전에 원문 페이지와 웹 출처를 사람이 다시 대조하고, 동일 조건의 재현 실험을 수행해야 한다. "
        "재현 실험에는 모델과 라이브러리 버전, 하드웨어 구성, 입력 분포, 동시 요청 수, warm-up 방식, 캐시 정책, 측정 구간을 명시해야 한다. "
        "공개 자료에서 확인되지 않는 비용과 운영 난이도는 공급자 자료나 PoC 로그로 보완해야 하며, 단일 실험 결과를 일반적인 우열 판단으로 확대하지 않아야 한다. "
        "추가 자료가 확보되면 기존 결론을 그대로 유지하기보다 주장별 근거 연결과 조건을 다시 점검해야 한다. "
        "이 한계 때문에 본 보고서는 기술 선택을 확정하는 문서가 아니라, 공개 근거를 기준으로 비교 가능한 항목과 추가 검증이 필요한 항목을 구분한 사전 평가로 보는 것이 적절하다."
    )


def ensure_section_depth(text: str, name: str, minimum: int) -> str:
    """짧은 절을 사실 추가 없이 방법론·한계 설명으로 보완합니다."""
    value = str(text or "").strip()
    if len(value) >= minimum and value.count("\n\n") >= 1:
        return value
    fallback = _depth_fallback(name)
    if not value or value == "공개 정보 부족":
        return fallback
    return f"{value}\n\n{fallback}"


def has_usable_analysis(value) -> bool:
    """오류 객체와 빈 하위 결과를 정상 분석 결과로 통과시키지 않습니다."""
    if not isinstance(value, dict) or not value or value.get("parse_error"):
        return False
    meaningful = False
    for item in value.values():
        if isinstance(item, dict):
            if not has_usable_analysis(item):
                return False
            meaningful = True
        elif item not in ({}, [], "", None):
            meaningful = True
    return meaningful


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
        if not has_usable_analysis(state.get(key)):
            missing.append(f"{key} 누락 또는 오류")
            retry_targets.append(target)

    available_ids = {item.get("evidence_id") for item in state.get("references", [])}
    used_ids = set()
    for key in required:
        used_ids.update(collect_evidence_ids(state.get(key, {})))
    valid_used_ids = {
        item for item in used_ids if VALID_EVIDENCE_ID.fullmatch(item)
    }
    unknown_ids = sorted(
        item for item in valid_used_ids if item not in available_ids
    )
    if unknown_ids:
        missing.append(f"존재하지 않는 evidence_id: {unknown_ids[:10]}")
        # 잘못된 인용이 나온 분석 영역만 재실행합니다. 전체 Worker 재실행은
        # 동적 재작업 요구사항을 깨고 검색 비용도 불필요하게 늘립니다.
        for key, target in required.items():
            # synthesis는 새 근거를 검색하지 않고 조합만 하므로, 잘못된 ID는
            # 보고서 단계에서 문단을 제외하고 synthesis 재호출은 하지 않습니다.
            if target != "synthesis" and collect_evidence_ids(state.get(key, {})) & set(unknown_ids):
                retry_targets.append(target)

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
    reference_index = [
        {
            "evidence_id": item.get("evidence_id"),
            "source_type": item.get("source_type"),
            "title": item.get("title"),
            "file_name": item.get("file_name"),
            "page": item.get("page"),
            "url": item.get("url"),
            "source_tier": item.get("source_tier"),
            "source_category": item.get("source_category"),
            "publisher": item.get("publisher"),
            "is_independent": item.get("is_independent"),
        }
        for item in state.get("references", [])
    ]
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
        "quality_feedback": state.get("quality_evaluation", {}),
        "reference_index": reference_index,
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
- 각 핵심 주장 또는 문단 끝에 입력 분석 결과의 evidence_id를 대괄호로 표시하세요. 예: [rag-xxxxxxxxxxxx]. 새로운 ID를 만들지 마세요.
- 허용된 evidence_id는 아래 `reference_index`에 있는 값뿐입니다. 목록에 없는 ID를 추측·수정·생성하지 마세요.
- 보고서 본문에 표시한 evidence_id는 최종 REFERENCE와 일대일로 연결되어야 하며, 근거가 없는 문장은 '공개 정보 부족'으로 표시하세요.
- 단, 4.1 TRL 절에서는 각 기술의 판정 이유와 근거 출처의 제목·페이지 또는 웹 출처명을 문장으로 명시하세요.
- 시장성 절과 이해관계자 절은 수집된 웹 근거를 반영하세요. 웹 근거가 없을 때만 공개 정보 부족이라고 쓰세요.
- 입력에 없는 수치, 기업 도입 사례, 시장 반응, 운영 결과를 추론해 사실처럼 쓰지 마세요.
- 직접 근거가 없으면 '공개 정보 부족'이라고 쓰고, 무엇이 부족한지와 판단에 미치는 영향을 설명하세요.
- '상용화 가능성이 높다', '비용 효율적이다', '효과적이다'처럼 전망이나 우열을 단정하는 표현은 직접 근거가 있을 때만 사용하세요. 직접 근거가 없으면 '공개 정보 부족', '잠정 해석', '추가 검증 필요'로 표현하세요.
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
- trl: 각 기술에 대해 확인된 실험 환경, 프로토타입·코드·시스템 통합·공개 배포·실제 운영 근거를 나누어 TRL을 추정하세요. DeepSeek-V2처럼 공개 출시·서비스 사용 근거가 있으면 TRL 6으로 고정하지 말고 TRL 8~9 가능성을 검토하세요. ITME처럼 FPGA 프로토타입과 관련 환경 시연만 있으면 TRL 6으로 설명하세요. 모델 전체의 배포 근거와 MLA 자체의 근거를 구분하고, 각 판정에 사용한 출처 제목·페이지 또는 웹 출처명을 본문에 명시하세요.
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
    section_minimums = {name: 650 for name in report_sections}
    # 시사점과 한계는 보고서 평가에서 빠지기 쉬운 부분이므로 별도 최소 분량을 둡니다.
    section_minimums.update({"implications": 900, "limitations": 900})
    short_sections = {
        name: report_data.get(name, "")
        for name in report_sections
        if len(str(report_data.get(name, ""))) < section_minimums[name]
        or str(report_data.get(name, "")).count("\n\n") < 1
    }
    if short_sections:
        expansion_prompt = f"""
    다음 보고서 절 초안은 기술평가 보고서로서 설명과 근거가 부족합니다.
    입력된 분석 결과와 기존 evidence_id만 사용해 각 절을 연구자·데이터센터 의사결정자가 읽을 수 있는 본문으로 확장하세요.
    모든 문자열은 한국어로 작성하고, JSON 키는 입력 키를 그대로 유지하세요.
    각 절은 최소 700자, 2개 이상의 문단으로 작성하세요.
    각 문단은 관찰 또는 주장, 근거, 해석, 판단의 한계 순서로 전개하세요.
    새로운 수치·사례·기업 반응을 만들지 말고, 각 핵심 주장 또는 문단 끝에 기존 evidence_id를 대괄호로 표시하세요.
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

    # 최종 본문은 등록된 ID만 사용합니다. 모델이 만든 미등록 ID는 문단 단위로 제외합니다.
    available_ids = {
        item.get("evidence_id")
        for item in state.get("references", [])
        if VALID_EVIDENCE_ID.fullmatch(str(item.get("evidence_id", "")))
    }
    for name in ["summary", *report_sections]:
        report_data[name] = exclude_unregistered_evidence(
            str(report_data.get(name, "")), available_ids
        )
    for name in ("implications", "limitations"):
        report_data[name] = ensure_section_depth(
            report_data[name], name, section_minimums[name]
        )

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
    # 최종 보고서 본문에 실제 표시된 ID만 REFERENCE로 연결합니다.
    used_ids = set(re.findall(r"(?:rag|web)-[a-f0-9]{12}", body))
    # 검증까지는 내부 ID를 유지하고 최종 출력 직전에만 출처 단위 숫자로 바꿉니다.
    evidence_to_number, references, citation_map = build_numbered_references(
        state.get("references", []), used_ids
    )
    for evidence_id, number in evidence_to_number.items():
        body = body.replace(f"[{evidence_id}]", f"[{number}]")
    # 같은 출처의 여러 청크를 연속으로 인용한 경우 중복 숫자를 정리합니다.
    body = re.sub(r"\[(\d+)\](?:\s+\[\1\])+", r"[\1]", body)
    report = body + "\n\n# REFERENCE\n\n" + references
    reference_count = sum(1 for line in references.splitlines() if line.startswith("- ["))
    print(f"[REFERENCE] 고유 출처 {reference_count}개 (누적 evidence {len(state.get('references', []))}건)")
    print("[6/6] 보고서 생성 완료")
    return {"report": report, "citation_map": citation_map}
