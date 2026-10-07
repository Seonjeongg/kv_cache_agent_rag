"""종합 평가, 근거 검증 Judge, 보고서 생성. (담당: 종합·검증·보고서)"""
from __future__ import annotations

import json
import re

from agents.technical_research import TERM_GLOSSARY
from config import FAST_MODE, MAX_RETRIES, REPORT_NUM_PREDICT
from evidence import build_numbered_references, collect_evidence_ids, compact_evidence
from llm import ask_json
from state import AgentState

VALID_EVIDENCE_ID = re.compile(r"^(?:rag|web)-[0-9a-f]{12}$")
EVIDENCE_TOKEN = re.compile(r"\b(?:rag|web)-[0-9a-f]{6,}\b")
SECTION_ANALYSIS = {
    "deepseek_overview": "technical_analysis", "trl": "trl_analysis",
    "itme_overview": "technical_analysis",
    "market": "market_analysis", "stakeholder": "stakeholder_analysis",
    "domain": "domain_analysis",
}
SECTION_EDITOR_GOALS = {
    "deepseek_overview": "1문단은 저차원 KV 공동 압축, 2문단은 decoupled RoPE가 행렬 흡수 문제를 어떻게 다루는지 설명한다. 원문 저자의 설명과 별도 운영 검증 필요를 구분한다.",
    "itme_overview": "1문단은 CXL-hybrid 메모리와 DMA 이동 원리, 2문단은 프로토타입 검증 범위와 호환성·대역폭의 확인 과제를 설명한다. '완벽하게', '보장', 무조건적 비용 절감은 쓰지 않는다.",
    "trl": "기술 원리를 반복하지 말고 성숙도 판단을 쓴다. 1문단은 DeepSeek-V2의 논문 실험·공개 배포와 MLA 구성요소의 실제 운영 검증을 구분하고, 입력에 운영 근거가 없으면 TRL 8~9 확정은 유보한다. 2문단은 ITME FPGA 프로토타입과 실제 운영 증거 부족을 근거로 TRL 6을 잠정 추정한다. 두 판정 모두 출처 제목과 페이지, 한계를 명시한다.",
    "market": "1문단은 DeepSeek-V2 공개 모델·구현 접근성과 실제 채택률 자료 부족, 2문단은 CXL 공급자의 공개 제품·생태계 자료와 ITME 자체 채택 증거 부족을 설명한다. 매개변수·벤치마크·훈련 비용 수치는 쓰지 않는다. DeepSeek-V2가 CXL을 필수로 요구한다거나 시장 수요가 증가한다고 단정하지 않는다.",
    "stakeholder": "1문단은 LLM·서비스 개발자의 모델 접근성·언어·환각 검증 요구, 2문단은 클라우드 운영자·메모리 공급자의 호환성·투자·지연 검증 요구를 쓴다. 각 요구는 공개 자료를 바탕으로 한 분석자의 잠정 해석이며 실제 인터뷰 결과가 아님을 표시한다. DeepSeek 조직 전체 비용·다른 모델 비용 수치는 쓰지 않는다.",
    "domain": "1문단은 MLA의 KV 메모리 절감 계층과 긴 입력·동시성·정확도 검증, 2문단은 ITME의 GPU-확장 메모리 이동·I/O 지연·호환성 검증을 쓴다. 모델 학습 비용은 서빙 비용 증거가 아니므로 생략한다. ITME 35.7%는 원문 실험의 CPU-offload 대비 수치이지 일반 운영 효과가 아님을 명시한다.",
}


def number_report_citations(text: str, evidence_to_number: dict[str, int]) -> str:
    """단일 인용과 [id1, id2] 묶음 인용을 모두 숫자 인용으로 바꾼다."""
    def replace_group(match):
        content = match.group(1)
        ids = EVIDENCE_TOKEN.findall(content)
        if not ids or EVIDENCE_TOKEN.sub("", content).strip(" ,;\t"):
            return match.group()
        if any(evidence_id not in evidence_to_number for evidence_id in ids):
            raise ValueError("숫자 인용으로 바꿀 수 없는 Evidence ID가 있습니다.")
        numbers = dict.fromkeys(evidence_to_number[evidence_id] for evidence_id in ids)
        return " ".join(f"[{number}]" for number in numbers)
    return re.sub(r"\[([^\]\n]+)\]", replace_group, text)


def exclude_unregistered_evidence(text: str, available_ids: set[str]) -> str:
    """미등록 인용이 붙은 문장은 제외하고 확인 가능한 문장은 남깁니다."""
    paragraphs = re.split(r"\n{2,}", str(text or "").strip())
    safe = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?。！？])\s+(?!\[)|\n+", paragraph)
        kept = []
        for sentence in sentences:
            if set(EVIDENCE_TOKEN.findall(sentence)) - available_ids:
                kept.append("공개 정보 부족: 출처와 연결할 수 없어 해당 주장은 제외하였다.")
            elif sentence.strip():
                kept.append(sentence.strip())
        safe.append(" ".join(kept))
    return "\n\n".join(item for item in safe if item).strip()


def _depth_fallback(name: str) -> str:
    if name == "background":
        return (
            "평가의 출발점은 KV cache를 단순한 메모리 절감 문제가 아니라 모델 구조, 메모리 계층, 입력 길이와 동시성에 함께 영향을 받는 시스템 문제로 보는 것이다. "
            "따라서 공개 자료의 기술 설명과 실험 수치를 실제 데이터센터 성능으로 바로 일반화하지 않고, 측정 조건과 적용 범위를 함께 기록해야 한다."
            "\n\n"
            "본 보고서는 기술 성숙도, 시장성, 이해관계자 요구, 도메인 적용성의 네 관점으로 두 접근을 검토한다. "
            "이 범위는 기술 원리만 확인하는 데서 끝나지 않고 도입 장벽과 운영 검증 과제까지 구분하기 위한 것이다."
        )
    if name == "selection":
        return (
            "두 기술을 함께 선정한 이유는 동일한 KV cache 병목을 모델 내부와 메모리 시스템이라는 서로 다른 계층에서 다루기 때문이다. "
            "이 차이는 비교의 범위를 넓혀 주지만, 실험 조건과 평가 단위가 다르므로 수치의 직접 대조에는 주의가 필요하다."
            "\n\n"
            "따라서 비교 결과는 우열 판정이 아니라 각 기술이 어떤 병목과 요구사항에 대응하는지, 그리고 어떤 추가 실험이 필요한지를 설명하는 데 사용한다."
        )
    if name == "comparison_conflicts":
        return (
            "두 기술의 비교는 동일한 목표를 서로 다른 계층에서 다루는지부터 구분해야 한다. "
            "DeepSeek-V2 MLA는 모델의 어텐션과 KV 표현 방식을 바꾸는 접근이고, ITME는 HBM과 확장 메모리 사이의 배치·이동 경로를 다루는 접근이다. "
            "따라서 메모리 사용량, 긴 컨텍스트, 동시 요청 처리, 데이터 이동 지연, 구현 복잡도는 비교 가능한 항목이지만 동일한 실험 조건에서 측정되지 않았다면 수치를 바로 대조할 수 없다. "
            "한 기술의 논문 수치가 다른 기술보다 크거나 작다는 사실만으로 전체적인 우열을 판단하는 것도 적절하지 않다. "
            "각 결과가 어떤 모델, 하드웨어, 입력 길이, 동시성, 소프트웨어 버전에서 측정됐는지를 함께 확인해야 한다."
            "\n\n"
            "두 접근은 경쟁 관계로만 볼 필요도 없다. 모델 내부의 KV 표현을 줄이는 방식과 외부 메모리 계층을 확장하는 방식은 동일한 시스템에서 서로 다른 병목을 완화할 가능성이 있지만, 실제 결합 가능성은 구현 호환성과 데이터 이동 비용을 별도로 검증해야 한다. "
            "비교의 핵심은 전체적인 우열을 정하는 것이 아니라 어떤 조건에서 어떤 병목이 남는지 확인하는 데 있다. "
            "후속 검증에서는 동일한 모델과 입력 집합을 사용해 HBM 사용량, TTFT, 처리량, tail latency, 정확도, 메모리 이동량을 같은 측정 절차로 기록해야 한다. "
            "그 결과가 확보되기 전까지는 공개 자료가 보여주는 장점과 실제 운영 효과를 구분해 해석해야 한다."
        )
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
        "등록되지 않은 근거 ID가 붙은 주장은 제외했으며, 이는 해당 주장이 거짓이라는 뜻이 아니라 현재 자료로 완전히 추적할 수 없다는 뜻이다."
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


def section_sources(state: AgentState, name: str) -> list[dict]:
    """절별 원문 발췌를 선별한다. 관련 없는 출처를 인용 수만 채우려고 넣지 않는다."""
    ids = collect_evidence_ids(state.get(SECTION_ANALYSIS[name], {}))
    if name == "trl":
        ids.update(collect_evidence_ids(state.get("technical_analysis", {})))
    candidates = [item for item in state.get("references", [])
                  if item.get("evidence_text") and item.get("evidence_id") in ids]
    if name in {"deepseek_overview", "itme_overview"}:
        # 분석에서 빠진 실제 검색 발췌도 읽는다. 목차는 원리의 근거로 쓰지 않는다.
        candidates = [item for item in state.get("references", [])
                      if item.get("evidence_text") and item.get("source_type") == "paper"
                      and not item["evidence_text"].lstrip().startswith("Contents")]
    if name == "trl":
        candidates = [item for item in state.get("references", [])
                      if item.get("source_type") == "paper" and item.get("evidence_text")
                      and any(term in item["evidence_text"].lower()
                              for term in ("prototype", "code", "released", "available"))
                      and not item["evidence_text"].lstrip().startswith("Contents")]
    if name in {"market", "stakeholder"}:
        candidates = [item for item in candidates if item.get("source_type") != "paper"]
    if name in {"deepseek_overview", "itme_overview"}:
        technology = "DeepSeek-V2 MLA" if name == "deepseek_overview" else "ITME"
        candidates = [item for item in candidates if item.get("technology") == technology]
    if name == "deepseek_overview":
        candidates.sort(key=lambda item: not any(
            term in item["evidence_text"].lower() for term in ("low-rank", "decoupled", "latent attention")
        ))
    if name in {"deepseek_overview", "itme_overview"}:
        return candidates[:4]
    # 먼저 서로 다른 출처를 하나씩, 이후 같은 출처의 추가 문단을 선택한다.
    grouped = {}
    for item in candidates:
        key = item.get("url") or item.get("file_name") or item.get("title")
        grouped.setdefault(key, []).append(item)
    selected = []
    for offset in range(2):
        selected.extend(items[offset] for items in grouped.values() if len(items) > offset)
    return selected[:8]


def grounded_paragraph_schema(ids: list[str]) -> dict:
    # Groundedness: 읽어 준 원문 ID만 선택하게 하고, 의미의 정확성은 별도 검토한다.
    return {
        "type": "object", "additionalProperties": False, "required": ["paragraphs"],
        "properties": {"paragraphs": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["text", "evidence_ids"],
            "properties": {
                "text": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string", "enum": ids}},
            },
        }}},
    }


def grounded_paragraph_text(result: dict, allowed_ids: set[str]) -> str:
    # 인용 수를 맞추려고 임의 출처를 붙이지 않고 모델이 고른 ID를 다시 검사한다.
    if not result.get("paragraphs"):
        raise ValueError("원문 대조 문단이 비어 있습니다.")
    paragraphs = []
    for paragraph in result.get("paragraphs", []):
        text = paragraph.get("text", "").strip()
        ids = list(dict.fromkeys(paragraph.get("evidence_ids", [])))
        if not text or not ids or set(ids) - allowed_ids:
            raise ValueError("원문 대조 문단의 본문 또는 근거 ID가 유효하지 않습니다.")
        # 본문/ID 필드가 중복돼도 출처 표시는 한 번만 조립한다.
        text = re.sub(r"\[[^\]\n]*(?:rag|web)-[0-9a-f]+[^\]\n]*\]", "", text).strip()
        paragraphs.append(f"{text} {' '.join(f'[{item}]' for item in ids)}")
    return "\n\n".join(paragraphs)


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
{json.dumps(payload, ensure_ascii=False)}
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

    requested = set(state.get("required_task_ids") or required.values())
    required = {key: target for key, target in required.items() if target in requested or target == "synthesis"}

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
            # 합성 결과만 잘못됐다면 조사 Worker 대신 합성 노드만 재실행한다.
            if collect_evidence_ids(state.get(key, {})) & set(unknown_ids):
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
        "quality_feedback": state.get("quality_evaluation", {}),
        "reference_index": reference_index,
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
    prompt = f"""
당신은 연구자와 데이터센터 의사결정자를 위한 중립적 기술평가 보고서를 작성합니다.
아래의 검증된 Agent 분석 결과만 사용해, 배경과 근거에서 판단과 시사점으로 이어지는 보고서 본문을 작성하세요.
작성 과정이나 생각은 출력하지 말고 지정된 JSON 객체만 반환하세요.

[공통 작성 원칙]
- 재생성 시 quality_feedback의 neutrality_matches에 표시된 표현을 다시 사용하지 마세요. 해당 주장과 근거를 검토해 조건부 설명이나 확인되지 않은 정보로 구분하세요.
- JSON 키를 제외한 모든 문자열 값은 반드시 한국어로 작성하세요.
- 영어 근거도 한국어로 해석하되, 기술명·고유명사·약어·논문 제목은 원문 표기를 허용합니다.
- 하나의 문단에서 사실, 해석, 판단을 섞지 말고 각각 구분해 서술하세요.
- 각 핵심 주장 또는 문단 끝에 입력 분석 결과의 evidence_id를 대괄호로 표시하세요. 예: [rag-xxxxxxxxxxxx]. 새로운 ID를 만들지 마세요.
- 허용된 evidence_id는 아래 `reference_index`에 있는 값뿐입니다. 목록에 없는 ID를 추측·수정·생성하지 마세요.
- 보고서 본문에 표시한 evidence_id는 최종 REFERENCE와 일대일로 연결되어야 하며, 근거가 없는 문장은 '공개 정보 부족'으로 표시하세요.
- 보고서 본문에서는 `우승`, `최고`, `압도적`, `최적의 선택`, `최적의 조합`, `최적의 솔루션`, `추천`, `상용화가 가속화`, `비용 효율적인`처럼 우열·추천·시장 전망을 단정하는 표현을 사용하지 마세요. 이를 부정하는 설명에서도 해당 표현을 반복하지 말고 '직접 비교에는 한계가 있다', '추가 검증이 필요하다'처럼 쓰세요.
- 단, 4.1 TRL 절에서는 각 기술의 판정 이유와 근거 출처의 제목·페이지 또는 웹 출처명을 문장으로 명시하세요.
- 시장성 절과 이해관계자 절은 수집된 웹 근거를 반영하세요. 웹 근거가 없을 때만 공개 정보 부족이라고 쓰세요.
- 각 조사 관점은 실제 참고한 서로 다른 두 출처의 근거를 유지하세요. 시장성과 이해관계자는 두 웹 도메인을 구분해 출처별 주장 범위와 부족한 검증을 설명하세요. 도메인 적용성에는 두 기술의 원 논문 근거를 각각 연결하세요. 관련 없는 출처를 분량이나 평가 통과를 위해 붙이지 마세요.
- 입력에 없는 수치, 기업 도입 사례, 시장 반응, 운영 결과를 추론해 사실처럼 쓰지 마세요.
- DeepSeek-V2 모델 전체의 학습 비용·벤치마크 결과를 MLA 단독의 효과로 귀속하지 마세요. CXL 일반 자료를 ITME 제품의 채택·상용화 증거로 사용하지 마세요.
- 장치 균형 손실·전문가 라우팅·로드 밸런싱은 DeepSeekMoE의 설명이며 MLA의 동작이나 제약으로 쓰지 마세요. MLA의 핵심은 저차원 KV 공동 압축과 분리된 RoPE입니다.
- 출처의 존재만으로 주장이 검증되는 것은 아닙니다. 입력에 직접 확인된 수치만 조건과 함께 쓰고, 출처가 뒷받침하지 않는 인과·도입·성숙도 판단은 유보하세요.
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
{json.dumps(payload, ensure_ascii=False)}
"""
    report_sections = [
        "summary", "background", "selection", "deepseek_overview", "itme_overview",
        "trl", "market", "stakeholder", "domain", "comparison_conflicts",
        "implications", "limitations",
    ]
    report_data = ask_json(
        "당신은 근거 기반의 중립적인 기술 평가 보고서 작성자입니다.",
        prompt,
        num_predict=REPORT_NUM_PREDICT,
        string_fields=report_sections,
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
        "summary", "background", "selection", "deepseek_overview", "itme_overview",
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
    # 모든 절을 한 번에 확장하면 출력 한도 때문에 뒷절이 잘릴 수 있다.
    # 입력 JSON은 자르지 않고, 부족한 절을 세 개씩 나눠 보완한다.
    section_names = list(short_sections)
    for offset in range(0, len(section_names), 3):
        batch_sections = {name: short_sections[name] for name in section_names[offset:offset + 3]}
        expansion_prompt = f"""
    다음 보고서 절 초안은 기술평가 보고서로서 설명과 근거가 부족합니다.
    입력된 분석 결과와 기존 evidence_id만 사용해 각 절을 연구자·데이터센터 의사결정자가 읽을 수 있는 본문으로 확장하세요.
    모든 문자열은 한국어로 작성하고, JSON 키는 입력 키를 그대로 유지하세요.
    각 절은 최소 700자, 2개 이상의 문단으로 작성하세요.
    각 문단은 관찰 또는 주장, 근거, 해석, 판단의 한계 순서로 전개하세요.
    새로운 수치·사례·기업 반응을 만들지 말고, 각 핵심 주장 또는 문단 끝에 기존 evidence_id를 대괄호로 표시하세요.
    원래 보고서와 동일한 중립성 기준을 유지하세요. '최적의 조합', '최적의 솔루션', '비용 효율적', '상용화가 가속', 특정 기술의 추천·우열 판정을 넣지 마세요.
    경제성은 이미 입증된 효과로 단정하지 말고 공개 자료의 조건과 추가 측정이 필요한 항목으로 설명하세요.
    초안이나 Agent 분석에 오류가 있어도 그대로 확대하지 마세요. 다음 원칙이 초안보다 우선합니다.
    - MLA는 저차원 KV 공동 압축과 decoupled RoPE입니다. 장치 균형 손실·전문가 라우팅·로드 밸런싱은 DeepSeekMoE이며 MLA의 원리나 제약으로 서술하지 마세요.
    - 42.5% 학습 비용 감소와 MMLU 점수는 DeepSeek-V2 모델 전체의 결과이며 MLA 단독의 인과 효과가 아닙니다.
    - MLA와 ITME는 서로 다른 계층입니다. 직접 경쟁 관계라고 단정하지 말고 결합 가능성은 검증되지 않은 가설로 구분하세요.
    - TRL은 본 보고서의 공개 근거 기반 잠정 추정입니다. 실제 운영 증거가 없으면 운영 검증이 완료됐다고 쓰지 마세요.
    - 시장·이해관계자 문장은 실제 조사된 반응과 분석자의 예상 요구를 구분하고, 일반 CXL 자료를 ITME 채택 증거로 쓰지 마세요.
    근거가 부족하면 단순히 문장을 반복하지 말고, 확인되지 않은 정보와 그로 인한 비교·판단의 한계를 설명하세요.

확장할 절 초안:
{json.dumps(batch_sections, ensure_ascii=False)}

참고할 분석 결과:
{json.dumps(payload, ensure_ascii=False)}
"""
        expanded = ask_json(
            "당신은 한국어 기술 평가 보고서 편집자입니다. 초안의 오류는 수정하고, MLA와 MoE를 구분하며 기술 성숙도와 운영 효과는 공개 근거 기반 잠정 해석으로 표시하세요.",
            expansion_prompt,
            num_predict=5000,
            string_fields=list(batch_sections),
        )
        for name in batch_sections:
            value = expanded.get(name)
            # 짧아졌더라도 잘못된 인과·기술 귀속을 고친 편집 결과를 버리지 않는다.
            if isinstance(value, str) and value.strip():
                report_data[name] = value

    # 분석 요약만 읽으면 기술 귀속이나 인용이 유실될 수 있어 핵심 절은 원문 발췌와 대조한다.
    for name in ("deepseek_overview", "itme_overview", "trl", "market", "stakeholder", "domain"):
        sources = section_sources(state, name)
        if not sources:
            continue
        source_ids = [item["evidence_id"] for item in sources]
        edited = ask_json(
            "당신은 원문 대조 편집자입니다. 입력 발췌가 뒷받침하는 사실만 쓰고 원문과 해석을 구분하세요.",
            f"""보고서의 {name} 절을 아래 원문 발췌로 다시 작성하세요. 2개 문단을 paragraphs 배열로 반환하세요.
이 절의 작업: {SECTION_EDITOR_GOALS[name]}
각 문단은 text(한국어 본문)와 evidence_ids(실제로 뒷받침하는 원문 ID 목록)로 구분하세요. text에는 ID를 직접 쓰지 마세요.
초안보다 원문이 우선입니다. 발췌에 없는 수치·운영 실적·기업 반응을 사실로 쓰지 마세요.
각 출처가 말하는 범위와 한계를 구분하고 ID는 evidence_ids 필드에만 쓰세요.
시장·관계자 절에서는 서로 다른 출처의 관찰을 구분하되 단순히 인용 개수를 채우려고 출처를 붙이지 마세요.
도메인은 두 원 논문의 적용 계층·실험 조건을 각각 설명하세요.
deepseek_overview는 DeepSeek-V2 MLA만, itme_overview는 ITME만 설명하세요. market/stakeholder/domain은 두 기술을 구분하세요.
웹 문서가 여러 DeepSeek 모델을 다루더라도 V3의 FP8·DualPipe를 V2의 기술로 쓰지 마세요. 시장·관계자 절에는 벤치마크·매개변수·학습 비용 수치를 반복하지 말고 공급자 공개 자료와 외부 관찰의 차이, 도입 시 검증할 요구사항을 설명하세요.
관계자 절의 두 문단은 LLM·클라우드 운영자와 메모리·인프라 공급자 관점으로 나누고, 실제 인터뷰가 아니라 공개 자료에서 도출한 잠정 요구임을 명시하세요.
원문 B 단위는 billion이며 236B는 2,360억, 21B는 210억입니다. 수치의 단위와 비교 기준을 원문 그대로 유지하고 불확실하면 수치를 생략하세요.
MLA는 low-rank KV 압축과 decoupled RoPE입니다. 원래 RoPE의 행렬 흡수 문제를 해결하려는 설계를 여전히 해결되지 않은 MLA의 결함으로 서술하지 마세요. MoE 부하 분산은 별도입니다.
TRL은 실험/공개 배포/운영 검증을 구분한 잠정 추정이며, 원문에서 실제 운영이 확인되지 않으면 유보하세요.
승자·최적 선택·추천을 정하지 마세요. 일반 CXL 생태계를 ITME 채택 증거로 쓰지 마세요.
원문 발췌:
{compact_evidence(sources, max_chars=12000)}
반환 형식: {{"paragraphs": [{{"text": "본문", "evidence_ids": ["원문 ID"]}}]}}""",
            num_predict=2200, json_schema=grounded_paragraph_schema(source_ids),
        )
        corrected = grounded_paragraph_text(edited, set(source_ids))
        report_data[name] = corrected

    # 본문과 요약 모두 같은 인용 검사를 적용한다.
    available_ids = {
        item.get("evidence_id")
        for item in state.get("references", [])
        if VALID_EVIDENCE_ID.fullmatch(str(item.get("evidence_id", "")))
    }
    for name in report_sections:
        report_data[name] = exclude_unregistered_evidence(
            str(report_data.get(name, "")), available_ids
        )
    for name in (
        "background", "selection", "comparison_conflicts", "implications", "limitations",
    ):
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
    body = number_report_citations(body, evidence_to_number)
    # 같은 출처의 여러 청크를 연속으로 인용한 경우 중복 숫자를 정리합니다.
    body = re.sub(r"\[(\d+)\](?:\s+\[\1\])+", r"[\1]", body)
    report = body + "\n\n# REFERENCE\n\n" + references
    reference_count = sum(1 for line in references.splitlines() if line.startswith("- ["))
    print(f"[REFERENCE] 고유 출처 {reference_count}개 (누적 evidence {len(state.get('references', []))}건)")
    print("[6/6] 보고서 생성 완료")
    return {"report": report, "citation_map": citation_map}
