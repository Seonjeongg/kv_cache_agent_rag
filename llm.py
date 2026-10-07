"""OpenAI API 호출 Helper: JSON/텍스트 응답과 JSON 복구."""
from __future__ import annotations

import json
import re

from json_repair import repair_json

from config import JSON_NUM_PREDICT, LLM_MODEL, REPORT_NUM_PREDICT, openai_client


def strip_model_reasoning(content: str) -> str:
    """Qwen 계열이 반환한 <think> 내부 추론을 최종 결과에서 제거합니다."""
    if "</think>" in content:
        content = content.split("</think>", 1)[1]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return content.strip()


def report_string_format(fields: list[str]) -> dict:
    """보고서 본문은 중첩 분석 객체가 아닌 절별 문자열로 제한한다."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "report_sections", "strict": True,
            "schema": {
                "type": "object", "additionalProperties": False,
                "properties": {field: {"type": "string"} for field in fields},
                "required": fields,
            },
        },
    }


def ask_json(
    system_prompt: str, user_prompt: str, num_predict: int | None = None,
    string_fields: list[str] | None = None,
    json_schema: dict | None = None,
) -> dict:
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY를 설정한 뒤 LLM 노드를 실행하세요.")
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\n모든 자연어 문자열 값은 반드시 한국어로 작성하세요. "
                    + "JSON 키는 요청된 형식을 유지하고 결과 JSON만 출력하세요. "
                    + "evidence_id는 입력 근거 목록에 실제로 존재하는 값을 그대로 복사하고, "
                    + "새로운 ID나 '공개 정보 부족' 같은 문구를 evidence_ids에 넣지 마세요."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        # 보고서 형식만 엄격히 제한한다. 기존 조사 Agent의 JSON 계약은 유지한다.
        response_format=(
            {"type": "json_schema", "json_schema": {
                "name": "grounded_paragraphs", "strict": True, "schema": json_schema,
            }} if json_schema else
            report_string_format(string_fields) if string_fields else {"type": "json_object"}
        ),
        temperature=0.1,
        max_tokens=num_predict or JSON_NUM_PREDICT,
    )
    content = strip_model_reasoning(response.choices[0].message.content or "")
    json_start = content.find("{")
    json_end = content.rfind("}")
    candidate = content[json_start:json_end + 1] if json_start >= 0 and json_end > json_start else content

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    except json.JSONDecodeError as decode_error:
        # 로컬 소형 모델의 닫히지 않은 괄호·따옴표 등 경미한 JSON 오류를 복구합니다.
        try:
            repaired = repair_json(candidate, return_objects=True)
            if isinstance(repaired, dict) and repaired:
                print(f"[JSON] 형식 오류 자동 복구: {type(decode_error).__name__}")
                return repaired
        except Exception as repair_error:
            print(f"[JSON] 자동 복구 실패: {type(repair_error).__name__}: {repair_error}")
        print(f"[JSON] 파싱 실패, 출력 길이={len(content)}자, 앞부분={content[:160]!r}")
        return {"parse_error": True, "raw_output": content}


def ask_text(system_prompt: str, user_prompt: str, num_predict: int | None = None) -> str:
    if openai_client is None:
        raise RuntimeError("OPENAI_API_KEY를 설정한 뒤 LLM 노드를 실행하세요.")
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt + "\n모든 답변은 반드시 한국어로 작성하세요.",
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=num_predict or REPORT_NUM_PREDICT,
    )
    return strip_model_reasoning(response.choices[0].message.content or "")
