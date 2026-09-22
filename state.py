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
        key = item.get("evidence_id") or item.get("url") or json.dumps(item, sort_keys=True)
        merged[key] = item
    return list(merged.values())


class AgentState(TypedDict, total=False):
    input_request: str
    selected_technologies: dict
    selection_reason: str
    domain: str

    technical_analysis: dict
    trl_analysis: dict
    market_analysis: dict
    stakeholder_analysis: dict
    domain_analysis: dict
    synthesis: dict

    validation_result: Literal["pass", "retry", "pass_with_limitations"]
    missing_evidence: list[str]
    retry_targets: list[str]
    retry_count: int

    report: str
    references: Annotated[list[dict], merge_references]
    errors: Annotated[list[str], add]
