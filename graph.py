"""Orchestrator-Workers graph with dynamic fan-out, fallback, and quality loop."""
from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.domain_evaluation import domain_evaluation_agent
from agents.market_evaluation import market_evaluation_agent
from agents.stakeholder_evaluation import stakeholder_evaluation_agent
from agents.synthesis import report_generation_agent, synthesis_agent, validation_judge
from agents.technical_research import technical_research_agent
from agents.technology_selection import technology_selection_agent
from config import MAX_RETRIES
from evidence import collect_evidence_ids
from state import AgentState


MAX_STEPS = 4
TASKS: dict[str, dict[str, Any]] = {
    "technical_research": {
        "objective": "MLA와 ITME의 원리·성능·한계·TRL을 논문 근거로 조사",
        "output_key": "technical_analysis, trl_analysis",
        "keywords": ("기술", "성숙도", "trl", "원리", "성능", "한계", "technical", "maturity"),
    },
    "market_evaluation": {
        "objective": "두 기술의 시장성·생태계·채택·도입 장벽을 조사",
        "output_key": "market_analysis",
        "keywords": ("시장", "시장성", "채택", "생태계", "market", "adoption"),
    },
    "stakeholder_evaluation": {
        "objective": "클라우드 사업자·개발자·운영자·하드웨어 업체의 기대와 우려를 조사",
        "output_key": "stakeholder_analysis",
        "keywords": ("이해관계자", "운영자", "개발자", "stakeholder", "operator"),
    },
    "domain_evaluation": {
        "objective": "데이터센터·클라우드 LLM 서빙 적용성을 동일 기준으로 조사",
        "output_key": "domain_analysis",
        "keywords": ("도메인", "적용성", "서빙", "데이터센터", "domain", "serving"),
    },
}
TASK_RUNNERS = {
    "technical_research": technical_research_agent,
    "market_evaluation": market_evaluation_agent,
    "stakeholder_evaluation": stakeholder_evaluation_agent,
    "domain_evaluation": domain_evaluation_agent,
}
TASK_OUTPUT_KEYS = {
    "technical_research": ("technical_analysis", "trl_analysis"),
    "market_evaluation": ("market_analysis",),
    "stakeholder_evaluation": ("stakeholder_analysis",),
    "domain_evaluation": ("domain_analysis",),
}
REQUIRED_REPORT_HEADINGS = (
    "# SUMMARY",
    "# 1. 분석 배경",
    "# 2. 기술 선정",
    "# 3. 기술 개요",
    "# 4. 관점별 평가",
    "# 6. 시사점",
    "# 7. 분석의 한계",
    "# REFERENCE",
)
PERSPECTIVE_TASKS = {
    "technical_research": ("technical_analysis", "trl_analysis"),
    "market_evaluation": ("market_analysis",),
    "stakeholder_evaluation": ("stakeholder_analysis",),
    "domain_evaluation": ("domain_analysis",),
}
REPORT_PERSPECTIVE_HEADINGS = {
    "technical_research": ("## 3.1 DeepSeek-V2 MLA", "## 3.2 ITME", "## 4.1 기술 성숙도 및 TRL"),
    "market_evaluation": ("## 4.2 시장성",),
    "stakeholder_evaluation": ("## 4.3 이해관계자",),
    "domain_evaluation": ("## 4.4 데이터센터·클라우드 적용성",),
}
REPORT_DEPTH_REQUIREMENTS = {
    "# 6. 시사점": 900,
    "# 7. 분석의 한계": 900,
}
NEUTRALITY_PATTERNS = (
    r"승자(?:다|이다|입니다|가|는)|우승|최고|압도적|명백히\s*우월|강력\s*추천|무조건\s*추천",
    r"가장\s*(?:우수|좋|적합)|최적(?:의|인)?\s*선택|추천(?:한다|함|하는 것이)",
    r"(?:MLA|ITME).{0,24}(?:더\s*우수|우월|최적(?:의|인)?\s*선택|추천)",
    r"최적(?:의|인)?\s*(?:조합|솔루션)",
    r"상용화가\s*가속",
    r"비용\s*효율(?:적|성을)",
)


def _event(node: str, event_type: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "type": event_type,
        "message": message,
        **extra,
    }


def _requested_task_ids(state: AgentState) -> list[str]:
    request = str(state.get("input_request", "")).lower()
    requested = [
        task_id for task_id, spec in TASKS.items()
        if any(keyword in request for keyword in spec["keywords"])
    ]
    return requested or list(TASKS)


def _task_plan(state: AgentState) -> list[dict[str, Any]]:
    retry_targets = set(state.get("retry_targets", []))
    # 관점이 명시된 질문이면 해당 Worker만 계획하고, 일반 질문이면 전체 관점을 조사한다.
    task_ids = list(retry_targets) if retry_targets else _requested_task_ids(state)
    task_ids = [task_id for task_id in TASKS if task_id in task_ids]
    selected = state.get("selected_technologies", {})
    technology_text = ", ".join(spec.get("name", side) for side, spec in selected.items())
    return [
        {
            "subtask_id": f"{task_id}-attempt-{int(state.get('retry_count', 0)) + 1}",
            "task_id": task_id,
            "agent": task_id,
            "objective": f"{spec['objective']} (대상: {technology_text})",
            "request": state.get("input_request", ""),
            "status": "pending",
            "attempt": int(state.get("retry_count", 0)) + 1,
        }
        for task_id, spec in TASKS.items()
        if task_id in task_ids
    ]


def orchestrator_plan_node(state: AgentState) -> dict[str, Any]:
    step = int(state.get("step_count", 0)) + 1
    if step > int(state.get("max_steps", MAX_STEPS)):
        return {
            "status": "warning",
            "plan": [],
            "decision_log": [_event("orchestrator", "termination", "step 상한 도달")],
        }
    plan = _task_plan(state)
    required_task_ids = state.get("required_task_ids") or [task["task_id"] for task in plan]
    task_ids = [task["task_id"] for task in plan]
    print(f"[PLAN] {len(plan)}개 subtask 결정: {', '.join(task_ids) or '없음'}")
    return {
        "plan": plan,
        "required_task_ids": required_task_ids,
        "plan_reason": "재시도 대상 우선" if state.get("retry_targets") else "input_request의 관점 키워드 기반 동적 분할",
        "step_count": step,
        "status": "running",
        "decision_log": [_event(
            "orchestrator",
            "plan",
            f"실행 시점에 {len(plan)}개 subtask를 구조화",
            task_ids=[task["task_id"] for task in plan],
            retry_targets=state.get("retry_targets", []),
        )],
    }


def fan_out_or_warn(state: AgentState):
    if state.get("status") == "warning":
        return "warning"
    # 계획에 포함된 수만큼만 Send를 생성한다: 고정 fan-out이 아니다.
    plan = state.get("plan", [])
    print(f"[FAN-OUT] {len(plan)}개 Worker 실행: {', '.join(task['task_id'] for task in plan) or '없음'}")
    return [Send("worker", {"current_task": task}) for task in plan]


def _usable(value: Any) -> bool:
    if not isinstance(value, dict) or not value or value.get("parse_error"):
        return False
    return any(value.values())


def _remove_unregistered_ids(value: Any, allowed_ids: set[str]) -> Any:
    """Worker 결과에서 실제 reference에 없는 ID만 제거합니다."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if key == "evidence_ids" and isinstance(item, list):
                cleaned[key] = [evidence_id for evidence_id in item if evidence_id in allowed_ids]
            elif key == "evidence_id" and isinstance(item, str):
                cleaned[key] = item if item in allowed_ids else ""
            else:
                cleaned[key] = _remove_unregistered_ids(item, allowed_ids)
        return cleaned
    if isinstance(value, list):
        return [_remove_unregistered_ids(item, allowed_ids) for item in value]
    return value


def _report_section_text(report: str, heading: str) -> str:
    """지정된 보고서 절만 추출해 절별 근거 연결을 검사합니다."""
    match = re.search(rf"^{re.escape(heading)}\s*$", report, flags=re.MULTILINE)
    if not match:
        return ""
    remainder = report[match.end():]
    next_heading = re.search(r"^#{1,3}\s+", remainder, flags=re.MULTILINE)
    return remainder[:next_heading.start()] if next_heading else remainder


def _report_evidence_ids(
    report: str,
    headings: tuple[str, ...] = (),
    citation_map: dict[str, list[str]] | None = None,
) -> set[str]:
    text = "\n".join(_report_section_text(report, heading) for heading in headings) if headings else report.split("# REFERENCE", 1)[0]
    evidence_ids = set(re.findall(r"(?:rag|web)-[a-f0-9]{12}", text))
    for number in re.findall(r"\[(\d+)\]", text):
        evidence_ids.update((citation_map or {}).get(number, []))
    return evidence_ids


def _source_domain(item: dict) -> str | None:
    domain = urlparse(item.get("url") or "").netloc.lower().split(":")[0]
    return domain.removeprefix("www.") or None


def _source_category(item: dict) -> str:
    if item.get("source_category"):
        return str(item["source_category"])
    # 이전 State와 직접 만든 테스트 fixture도 논문은 원 연구로 해석한다.
    return "primary_research" if item.get("source_type") == "paper" else "unclassified"


def _report_depth_ok(report: str) -> bool:
    return all(
        len(_report_section_text(report, heading).strip()) >= minimum
        and _report_section_text(report, heading).count("\n\n") >= 1
        for heading, minimum in REPORT_DEPTH_REQUIREMENTS.items()
    )


def worker_node(state: AgentState) -> dict[str, Any]:
    task = state.get("current_task", {})
    task_id = task.get("task_id")
    runner = TASK_RUNNERS.get(task_id)
    print(f"[WORKER] 시작: {task.get('subtask_id', task_id)}")
    if runner is None:
        return {
            "errors": [f"worker: unknown task {task_id}"],
            "task_results": [{"task_id": task_id, "status": "failed", "error": "unknown task"}],
            "decision_log": [_event("worker", "fallback", f"알 수 없는 task 제외: {task_id}")],
        }

    try:
        result = runner(state)
        allowed_ids = {
            item.get("evidence_id")
            for item in ((state.get("references") or []) + (result.get("references") or []))
            if item.get("evidence_id")
        }
        invalid_ids = collect_evidence_ids(result) - allowed_ids
        if invalid_ids:
            result = _remove_unregistered_ids(result, allowed_ids)
        failed = bool(result.get("errors")) or any(
            not _usable(result.get(key)) for key in TASK_OUTPUT_KEYS[task_id]
        )
        status = "failed" if failed else "success"
        message = (
            f"{task_id} 실패: 결과를 합성하되 해당 관점은 제한사항으로 표시"
            if failed
            else f"{task_id} 완료"
        )
        result["task_results"] = [{
            "task_id": task_id,
            "agent": task_id,
            "status": status,
            "attempt": int(task.get("attempt", 0)) + 1,
            "output_keys": list(TASK_OUTPUT_KEYS[task_id]),
        }]
        result["decision_log"] = [_event("worker", "fallback" if failed else "worker", message, task_id=task_id)]
        if invalid_ids:
            result["decision_log"].append(_event(
                "worker", "evidence_filter",
                "등록된 reference가 없는 evidence_id를 분석 결과에서 제외",
                task_id=task_id,
                excluded_ids=sorted(invalid_ids),
            ))
        print(f"[WORKER] {'실패' if failed else '완료'}: {task.get('subtask_id', task_id)}")
        return result
    except Exception as error:  # worker failure policy: continue with limitations
        return {
            "errors": [f"{task_id}: {type(error).__name__}: {error}"],
            "task_results": [{
                "task_id": task_id,
                "agent": task_id,
                "status": "failed",
                "attempt": int(task.get("attempt", 0)) + 1,
                "error": f"{type(error).__name__}: {error}",
            }],
            "decision_log": [_event(
                "worker", "fallback",
                f"{task_id} 예외 발생; 계속 진행하고 보고서에 한계 기록",
                task_id=task_id,
            )],
        }


def route_after_validation(state: AgentState) -> Literal["retry", "synthesis", "report"]:
    if state.get("validation_result") != "retry":
        return "report"
    targets = set(state.get("retry_targets", []))
    return "synthesis" if targets == {"synthesis"} else "retry"


def quality_evaluator_node(state: AgentState) -> dict[str, Any]:
    report = state.get("report", "")
    references = state.get("references", [])
    available_ids = {item.get("evidence_id") for item in references}
    # 분석 State의 전체 evidence가 아니라 최종 보고서 본문에 실제 표시된 ID만 사용합니다.
    citation_map = state.get("citation_map") or {}
    used_ids = _report_evidence_ids(report, citation_map=citation_map)
    active_task_ids = [
        task_id for task_id in (state.get("required_task_ids") or PERSPECTIVE_TASKS)
        if task_id in PERSPECTIVE_TASKS
    ]
    perspective_used_ids = {
        task_id: _report_evidence_ids(
            report, REPORT_PERSPECTIVE_HEADINGS[task_id], citation_map
        )
        for task_id in PERSPECTIVE_TASKS
    }
    evidence_by_id = {item.get("evidence_id"): item for item in references}
    source_by_id = {
        evidence_id: (
            f"url:{item.get('url')}" if item.get("url") else
            f"file:{item.get('file_name')}" if item.get("file_name") else
            f"title:{item.get('title')}"
        )
        for evidence_id, item in evidence_by_id.items()
    }
    perspective_sources = {
        task_id: sorted({source_by_id[evidence_id] for evidence_id in ids if evidence_id in source_by_id})
        for task_id, ids in perspective_used_ids.items()
    }
    source_counts = Counter(source_by_id[evidence_id] for evidence_id in used_ids if evidence_id in source_by_id)
    max_source_share = max(source_counts.values(), default=0) / max(sum(source_counts.values()), 1)
    bias_failures = [task_id for task_id in active_task_ids if len(perspective_sources[task_id]) < 2]
    if max_source_share > 0.5:
        dominant_source = source_counts.most_common(1)[0][0]
        bias_failures.extend(
            task_id for task_id in active_task_ids if dominant_source in perspective_sources[task_id]
        )
    bias_control = {
        "perspective_source_counts": {task_id: len(sources) for task_id, sources in perspective_sources.items()},
        "max_source_share": round(max_source_share, 3),
        "failed_perspectives": sorted(set(bias_failures)),
        "active_perspectives": active_task_ids,
        "passed": bool(used_ids) and bool(active_task_ids) and not bias_failures and max_source_share <= 0.5,
    }
    used_sources = {
        source_by_id[evidence_id]: evidence_by_id[evidence_id]
        for evidence_id in used_ids
        if evidence_id in evidence_by_id
    }
    category_counts = Counter(_source_category(item) for item in used_sources.values())
    market_domains = {
        domain for evidence_id in perspective_used_ids["market_evaluation"]
        if evidence_id in evidence_by_id
        for domain in [_source_domain(evidence_by_id[evidence_id])]
        if domain
    }
    stakeholder_domains = {
        domain for evidence_id in perspective_used_ids["stakeholder_evaluation"]
        if evidence_id in evidence_by_id
        for domain in [_source_domain(evidence_by_id[evidence_id])]
        if domain
    }
    core_ids = (
        perspective_used_ids["technical_research"]
        | perspective_used_ids["domain_evaluation"]
    )
    low_quality_core_claims = sorted(
        evidence_id for evidence_id in core_ids
        if evidence_id in evidence_by_id
        and evidence_by_id[evidence_id].get("source_tier") in {4, 5}
    )
    failed_reliability = []
    reliability_reasons = []
    if "technical_research" in active_task_ids:
        technical_primary = any(
            _source_category(evidence_by_id[evidence_id]) == "primary_research"
            for evidence_id in perspective_used_ids["technical_research"]
            if evidence_id in evidence_by_id
        )
        if not technical_primary:
            failed_reliability.append("technical_research")
            reliability_reasons.append("기술 핵심 수치를 뒷받침하는 원 논문 근거가 없습니다.")
    if "market_evaluation" in active_task_ids and len(market_domains) < 2:
        failed_reliability.append("market_evaluation")
        reliability_reasons.append("시장성 근거가 서로 다른 두 도메인에 미치지 못합니다.")
    if "stakeholder_evaluation" in active_task_ids and len(stakeholder_domains) < 2:
        failed_reliability.append("stakeholder_evaluation")
        reliability_reasons.append("이해관계자 근거가 서로 다른 두 도메인에 미치지 못합니다.")
    if low_quality_core_claims:
        affected = [
            task_id for task_id in ("technical_research", "domain_evaluation")
            if perspective_used_ids[task_id] & set(low_quality_core_claims)
        ]
        failed_reliability.extend(affected)
        reliability_reasons.append("4~5등급 출처가 핵심 기술 주장 또는 지표를 뒷받침합니다.")
    source_reliability = {
        "passed": not failed_reliability,
        "market_distinct_domains": len(market_domains),
        "stakeholder_distinct_domains": len(stakeholder_domains),
        "primary_source_count": category_counts["primary_research"],
        "official_source_count": category_counts["official"],
        "independent_source_count": category_counts["independent"],
        "low_quality_core_claims": low_quality_core_claims,
        "failed_perspectives": sorted(set(failed_reliability)),
        "reasons": reliability_reasons,
    }
    neutrality_failed = any(re.search(pattern, report, flags=re.IGNORECASE) for pattern in NEUTRALITY_PATTERNS)
    body_numbers = set(re.findall(r"\[(\d+)\]", report.split("# REFERENCE", 1)[0]))
    reference_numbers = set(re.findall(
        r"^- \[(\d+)\]", report.split("# REFERENCE", 1)[-1], flags=re.MULTILINE
    )) if "# REFERENCE" in report else set()
    numeric_connection_ok = (
        not body_numbers
        or body_numbers <= reference_numbers
        and body_numbers <= set(citation_map)
    )
    criteria = {
        "groundedness": bool(references) and bool(used_ids) and used_ids <= available_ids and numeric_connection_ok,
        "required_structure": all(heading in report for heading in REQUIRED_REPORT_HEADINGS),
        "section_depth": _report_depth_ok(report),
        "neutrality": not neutrality_failed,
        "bias_control": bias_control["passed"],
        "source_reliability": source_reliability["passed"],
        "perspective_coverage": all(
            all(_usable(state.get(key)) for key in PERSPECTIVE_TASKS[task_id])
            for task_id in active_task_ids
        ) and _usable(state.get("synthesis")),
        "reference_connection": bool(references) and bool(used_ids),
    }
    issues = [name for name, passed in criteria.items() if not passed]
    retry_targets = []
    if not criteria["perspective_coverage"]:
        for task_id in active_task_ids:
            keys = TASK_OUTPUT_KEYS[task_id]
            if any(not _usable(state.get(key)) for key in keys):
                retry_targets.append(task_id)
    if not criteria["bias_control"] and used_ids:
        retry_targets.extend(bias_control["failed_perspectives"] or list(PERSPECTIVE_TASKS))
    if not criteria["source_reliability"]:
        retry_targets.extend(source_reliability["failed_perspectives"])
    if not criteria["groundedness"]:
        # 본문 인용 누락·잘못된 ID는 조사 문제가 아니라 보고서 생성 문제입니다.
        # bias_control이 함께 실패해도 Worker를 재실행하지 않고 보고서를 다시 만듭니다.
        retry_targets = []

    retry_count = int(state.get("retry_count", 0))
    report_retry_count = int(state.get("report_retry_count", 0))
    if issues and retry_targets and retry_count < MAX_RETRIES:
        retry_phase = "workers"
        result = "retry"
        retry_count += 1
        status = "running"
    elif issues and not retry_targets and report_retry_count < MAX_RETRIES:
        retry_phase = "report"
        result = "report_retry"
        report_retry_count += 1
        status = "running"
    elif issues:
        result = "pass_with_limitations"
        retry_phase = "limit"
        status = "warning"
    else:
        result = "pass"
        retry_phase = "finish"
        status = "success"
    quality = {
        "result": result,
        "criteria": criteria,
        "issues": issues,
        "retry_targets": sorted(set(retry_targets)),
        "retry_phase": retry_phase,
        "method": "1안: 재현 가능한 deterministic rubric gate; 출처 다양성·인용 존재성 포함",
        "source_reliability": source_reliability,
    }
    return {
        "quality_evaluation": quality,
        "validation_result": "retry" if result == "retry" else "pass_with_limitations" if result == "pass_with_limitations" else "pass",
        "retry_targets": quality["retry_targets"],
        "retry_count": retry_count,
        "report_retry_count": report_retry_count,
        "status": status,
        "decision_log": [_event(
            "quality_evaluator", "quality_gate", f"보고서 품질 평가: {result}",
            criteria=criteria, issues=issues,
        )],
    }


def route_after_quality(state: AgentState) -> Literal["retry", "report_retry", "finish"]:
    result = state.get("quality_evaluation", {}).get("result")
    if result == "retry":
        return "retry"
    if result == "report_retry":
        return "report_retry"
    return "finish"


def warning_node(state: AgentState) -> dict[str, Any]:
    return {
        "status": "warning",
        "decision_log": [_event("warning", "termination", "제한사항을 남기고 종료")],
    }


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("technology_selection", technology_selection_agent)
    builder.add_node("orchestrator_plan", orchestrator_plan_node)
    builder.add_node("worker", worker_node)
    builder.add_node("synthesis", synthesis_agent)
    builder.add_node("validation", validation_judge)
    builder.add_node("report_generation", report_generation_agent)
    builder.add_node("quality_evaluator", quality_evaluator_node)
    builder.add_node("warning", warning_node)

    builder.add_edge(START, "technology_selection")
    builder.add_edge("technology_selection", "orchestrator_plan")
    builder.add_conditional_edges("orchestrator_plan", fan_out_or_warn, ["worker", "warning"])
    builder.add_edge("worker", "synthesis")
    builder.add_edge("synthesis", "validation")
    builder.add_conditional_edges(
        "validation",
        route_after_validation,
        {"retry": "orchestrator_plan", "synthesis": "synthesis", "report": "report_generation"},
    )
    builder.add_edge("report_generation", "quality_evaluator")
    builder.add_conditional_edges(
        "quality_evaluator",
        route_after_quality,
        {"retry": "orchestrator_plan", "report_retry": "report_generation", "finish": END},
    )
    builder.add_edge("warning", END)
    return builder.compile()


DEFAULT_REQUEST = (
    "DeepSeek-V2 MLA와 ITME를 데이터센터·클라우드 LLM 서빙 환경에서 "
    "TRL, 시장성, 이해관계자, 도메인 관점으로 중립적으로 비교 평가하라."
)


def initial_state(input_request: str | None = None) -> AgentState:
    return {
        "input_request": input_request or DEFAULT_REQUEST,
        "references": [],
        "citation_map": {},
        "errors": [],
        "decision_log": [],
        "trace_id": str(uuid.uuid4()),
        "status": "running",
        "step_count": 0,
        "max_steps": MAX_STEPS,
        "failure_policy": "continue_with_limitations",
        "retry_count": 0,
        "report_retry_count": 0,
        "retry_targets": [],
    }
