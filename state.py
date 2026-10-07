"""공통 Schema와 LangGraph State."""
from __future__ import annotations

import json
from operator import add
from typing import Annotated, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict


class Evidence(BaseModel):
    evidence_id: str
    agent: str
    technology: str
    source_type: str
    title: str
    claim: str
    evidence_text: str
    url: str | None = None
    file_name: str | None = None
    page: int | None = None
    chunk_id: str | None = None
    retrieval_score: float | None = None
    published_at: str | None = None


def merge_references(left: list[dict], right: list[dict]) -> list[dict]:
    """Loop와 병렬 실행에서 동일 출처가 중복되지 않게 합칩니다."""
    merged = {}
    for item in (left or []) + (right or []):
        if item.get("source_type") == "paper" and item.get("chunk_id"):
            key = ("paper", item.get("technology"), item["chunk_id"])
        elif item.get("source_type") == "web" and item.get("url"):
            key = ("web", item["url"])
        else:
            key = item.get("evidence_id") or json.dumps(item, sort_keys=True)
        merged[key] = item
    return list(merged.values())


def merge_task_results(left: list[dict], right: list[dict]) -> list[dict]:
    """Fan-out worker 결과를 task_id 기준으로 합치고 최신 시도를 유지합니다."""
    merged = {item.get("task_id"): item for item in (left or []) if item.get("task_id")}
    for item in right or []:
        if item.get("task_id"):
            merged[item["task_id"]] = item
    return list(merged.values())


class AgentState(TypedDict, total=False):
    # Payload: 조사 결과와 근거. Control: 실행 계획·라우팅·복구 상태.
    input_request: str
    selected_technologies: dict
    selection_reason: str
    domain: str

    plan: list[dict]
    plan_reason: str
    required_task_ids: list[str]
    current_task: dict
    task_results: Annotated[list[dict], merge_task_results]
    decision_log: Annotated[list[dict], add]
    trace_id: str
    status: Literal["running", "success", "warning", "failed"]
    step_count: int
    max_steps: int
    failure_policy: Literal["continue_with_limitations", "retry", "exclude"]

    technical_analysis: dict
    trl_analysis: dict
    market_analysis: dict
    stakeholder_analysis: dict
    domain_analysis: dict
    synthesis: dict

    validation_result: Literal["pass", "retry", "pass_with_limitations"]
    quality_evaluation: dict
    missing_evidence: list[str]
    retry_targets: list[str]
    retry_count: int
    report_retry_count: int

    report: str
    references: Annotated[list[dict], merge_references]
    errors: Annotated[list[str], add]
    runtime_metadata: dict
