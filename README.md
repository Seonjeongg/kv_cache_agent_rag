# Subject

KV cache 최적화 기술을 SW·HW 진영에서 하나씩 선정하고, 기술 성숙도·시장성·이해관계자·도메인 적용성을 평가하는 **Orchestrator-Workers 기반 Agentic RAG** 프로젝트입니다. 기술의 종합 우열이나 구매 추천이 아니라 공개 근거, 적용 조건, 제약과 추가 검증 과제를 정리합니다.

## Overview

- Objective: DeepSeek-V2 MLA와 ITME를 데이터센터·클라우드 LLM 서빙 관점에서 평가
- Pattern: Orchestrator-Workers. 서로 독립적인 관점 조사를 병렬 수행하고 Synthesizer가 취합하는 목적에 맞습니다.
- 동적 처리: 요청의 관점 키워드로 구조화된 `plan`을 만들고 그 길이만큼 `Send`를 생성합니다. 품질 미달 시에는 실패 관점만 다시 계획합니다. 고정된 네 갈래 연결이 아닙니다.
- Trade-off: 병렬 조사 시간을 줄일 수 있지만 동시 쓰기의 병합, 근거 중복과 출처 간 조건 차이를 관리해야 합니다. Worker 결과는 reducer로 합칩니다.
- 실패 정책: Worker 오류는 기록하고 제한사항과 함께 계속 진행합니다. 조사·보고서 재시도 상한을 넘으면 경고로 종료하며 성공으로 숨기지 않습니다.

## Selected Technologies

- SW: **DeepSeek-V2 MLA (Multi-head Latent Attention)** — 모델 내부의 저차원 KV 표현을 통한 메모리 절감 접근을 평가하기 위해 선정했습니다.
- HW: **ITME (Inference Tiered Memory Expansion)** — 분리된 CXL-Hybrid 메모리 계층을 통한 확장 접근을 평가하기 위해 선정했습니다.
- 팀이 비교 기술을 선정했으며 `technology_selection` 노드는 그 선택과 이유를 초기화합니다. 모델이 비교 대상을 자동 선정하는 구조는 아닙니다.

## Features

- 두 원 논문 PDF를 파싱하고 참고문헌 뒤 페이지를 제외한 뒤 900자 청크(120자 중복)로 색인합니다.
- 시장성·이해관계자는 Tavily 웹 검색을 사용하며 실패 시 DDGS/DuckDuckGo 경로로 대체합니다.
- 근거는 내부 `rag-...`·`web-...` ID로 검사합니다. 최종 보고서에만 같은 논문·URL을 `[1]`, `[2]`로 묶고, `citation_map`에 원본 ID를 보존합니다.
- 미등록 ID에 기대는 주장은 본문에서 제외합니다. ID가 존재한다고 주장 내용의 진실성까지 자동 입증되는 것은 아닙니다.
- 확증 편향 방지: 관점별 복수 출처, 단일 출처 집중도, 원 논문·공식·독립 출처 구분, 기업 발표와 외부 검증의 구별을 검사하거나 프롬프트에 명시합니다.
- 보고서 생성 뒤 별도 Quality Evaluator에서 Groundedness·중립성·편향 통제·관점 커버리지 및 출처 신뢰성을 판정하고 재작업합니다.
- 모델 전체의 성능을 MLA 단독 효과로, CXL 일반 자료를 ITME의 실제 채택 증거로 일반화하지 않도록 프롬프트에서 제한합니다.
- A4 PDF 생성 시 10장 이하와 필수 목차·숫자 인용 연결을 검사합니다.

## Tech Stack

| 구성 | 사용 기술 |
| --- | --- |
| Framework | LangGraph |
| LLM / Generator | OpenAI `gpt-4o-mini` (환경변수로 변경 가능) |
| LLM / Judge | 별도 LLM Judge 없음. 노션 1안에 해당하는 Python 규칙 기반 평가 |
| Retrieval | 로컬 ChromaDB, cosine 검색 + lexical rerank |
| Retrieval 평가 | 검증용 10문항, Hit@5·MRR. 결과: `outputs/retrieval_metrics.json` |
| Embedding | 오픈소스 `qwen3-embedding:0.6b`, Ollama 로컬 실행 |
| Parsing / PDF | PyMuPDF / Markdown + WeasyPrint |
| Tracing | LangSmith + 로컬 JSONL callback |

Ollama는 **문서·질의 임베딩**에 사용합니다. 보고서 생성은 OpenAI가 맡습니다. 문서와 검색 질의에 동일한 임베딩 모델을 쓰며, 모델 변경 시 색인을 다시 만들어야 합니다. 현재 10문항의 Hit@5는 0.50, MRR은 0.3833입니다. 검색 실패 5문항은 남아 있는 개선 과제입니다.

## Agents

| 노드 | 역할 |
| --- | --- |
| Orchestrator | 요청·재시도 대상을 읽어 계획 수립, 동적 fan-out |
| Technical Research | 논문 기반 원리·실험 조건·한계·TRL 추정 |
| Market Evaluation | 시장·채택·생태계·경제성 및 출처 구분 |
| Stakeholder Evaluation | 관계자 요구·우려와 근거 범위 |
| Domain Evaluation | 데이터센터·클라우드 적용성 |
| Synthesizer | 조사 결과의 일치·상충·보완 가능성 취합 |
| Validation | 필요한 분석 결과 및 원본 근거 ID 검사 |
| Report Generation | 본문 작성, 출처별 숫자 인용 변환 |
| Quality Evaluator | 생성된 보고서의 품질 판정과 후속 재작업 |
| Warning | 반복 상한 후 제한사항 종료 |

하위 Worker끼리 직접 호출하거나 결과를 전달하지 않습니다. 공통 State의 조사 결과를 병합한 뒤 Synthesizer가 읽습니다.

## State Schema

| 설계 항목 | 반영 내용 |
| --- | --- |
| 제어 vs 페이로드 | 제어: `plan`, `required_task_ids`, `retry_targets`, 횟수·상태. 페이로드: 관점 분석, `references`, `report`, `citation_map` |
| 관측성 위치 | 최소 결정·사유는 `decision_log`, 자세한 실행 경로는 LangSmith와 JSONL. 원문 프롬프트는 로컬 callback에 저장하지 않음 |
| 지속성 비용 | PDF·벡터 DB는 파일/Chroma에 저장. State에는 근거 발췌가 포함되므로 검색 수·중복 제거·반복 상한으로 누적량 제한 |
| 상관 | 동일 UUID `trace_id`를 State·root run_id·metadata·로컬 로그에 사용 |
| 재개/복구 | 실패 결과·대상·시도 수를 보존하여 한 실행 안에서 재시도. 프로세스 종료 후 복구하는 checkpointer는 미구현 |
| 동시 처리 | `merge_task_results`는 작업별 최신 결과, `merge_references`는 청크/근거 ID, 로그·오류는 누적 reducer 사용 |
| 종료 보장 | 계획 step 상한 4, 조사 재시도 2회, 보고서 재생성 2회, recursion limit 40. 상한 초과는 warning 종료 |

`source_tier`, `source_category`, `publisher`, `is_independent`를 Evidence에 보존합니다. 알려지지 않은 도메인은 임의로 낮은 등급을 부여하지 않고 `unclassified`로 남깁니다. 공식 저장소 플랫폼의 도메인만으로 게시자의 공신력까지 보장할 수 없으며, 원 논문도 독립된 외부 검증과 같지 않습니다. 현재 독립 출처 allowlist가 비어 있어 독립 검증 확보를 자동 보장하지 않습니다.

## Architecture

![Orchestrator-Workers 실행 구조](docs/architecture.png)

```text
START → technology_selection → orchestrator_plan
                                  ↓ Send(plan), N개
                                 worker
                                  ↓ reducer / fan-in
synthesis → validation → report_generation → quality_evaluator → END
    ↑           │                                  │
    └─ 합성 재시도                                  ├─ 대상 Worker 재계획
                                                    ├─ 보고서만 재생성
                                                    └─ 상한: warning 종료
```

## Directory Structure

```text
├── agents/       # 관점 조사·종합·보고서 프롬프트와 로직
├── data/         # 원 논문과 로컬 Chroma (Git 제외)
├── docs/         # 아키텍처, PDF 스타일, 제출 점검·실행 캡처
├── outputs/      # 보고서 MD/HTML/PDF, State JSON, 로컬 실행 로그
├── tests/        # API 호출 없이 실행하는 회귀·구조 검사
├── app.py        # 전체 실행 및 산출물 검증·저장
├── graph.py      # 계획·Send·Worker·품질 Loop
├── state.py      # Schema와 reducer
├── config.py     # 모델·검색·출처 분류·환경설정
├── rag.py        # PDF 파싱·색인·검색
├── evidence.py   # 근거 구성 및 숫자 참고문헌
├── search.py     # 웹 검색과 출처 분류
├── llm.py        # OpenAI 호출 및 JSON 응답 처리
├── tracing.py    # 실행 callback
├── evaluate.py   # Hit@5·MRR 측정
└── demo_fanout.py # 두 Worker 실제 실행 실험
```

프롬프트는 별도 `prompts/` 대신 해당 Agent 파일 내부에 관리합니다.

## Usage

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install pango          # macOS PDF 의존성
ollama pull qwen3-embedding:0.6b
# Ollama 앱/서버 실행 후 .env.example을 참고해 .env 설정
python tests/test_smoke.py
python app.py              # 제출용: 네 관점 전체 실행
python evaluate.py --reuse-index
python demo_fanout.py       # 별도 동적 fan-out 실험
```

`.env` 설정: `OPENAI_API_KEY`, 선택 `TAVILY_API_KEY`, `FAST_MODE=false`, `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=skala`. 키와 원문 실행 프롬프트는 공개 저장소에 넣지 않습니다.

`FAST_MODE=true`는 검색·출력·재시도를 축소하는 구조 확인용입니다. 제출은 false로 실행합니다. `app.py`가 색인을 재생성하는 동안 다른 프로세스에서 검색 평가를 실행하지 마세요. 전체 실행 후 기존 색인으로 평가합니다.

## LangSmith Tracing

`trace_id`로 State와 실제 LangSmith run을 연결합니다. OpenAI SDK는 `wrap_openai`로 감싸 하위 모델 호출을 추적하고, JSONL에는 노드 시작·종료, 계획·분기 사유, 품질 결과를 기록합니다.

- 제출용 캡처는 실제 LangSmith 실행 트리와 계획/품질 결과를 보여야 합니다.
- 긴 실행은 `tracing-1.png`, `tracing-2.png` 순서로 구분합니다.
- `demo_fanout.py`는 같은 그래프에서 두 Worker를 실행한 뒤 합성 전에 의도적으로 멈춥니다. 완료 보고서 실행과 구분합니다.
- 아키텍처 그림·로컬 로그 이미지는 실제 LangSmith 캡처를 대체하지 않습니다.
- 최신 통합 실행 결과와 가이드 검증은 `docs/submission_check_20261007.md`에 기록합니다.

## 평가 기준 대응

| 평가 요소 | 배점 | 구현·확인 위치 |
| --- | ---: | --- |
| 패턴 정합성 | 20 | `orchestrator_plan_node`, `fan_out_or_warn`, Worker fallback |
| 동적 동작 실증 | 20 | 실제 plan/Send trace, 2-Worker 실험 및 회귀 테스트 |
| State Schema | 20 | 위 7항목, `state.py`, callback/root UUID |
| 품질 평가 노드 | 15 | 생성 후 `quality_evaluator_node` + 조건부 Loop |
| 구조·모듈 분리 | 5 | 조정 계층 `graph.py` / 역할별 `agents/` |
| 재현성 | 10 | `app.py`, State·실행 메타, 종료·재시도 상한 |
| Output 보고서 | 10 | SUMMARY·1~7절·REFERENCE, PDF 10장 제한 |

평가는 **노션 1안(형식·연결성 규칙 검사)**입니다. 규칙 통과가 주장 의미의 정확성이나 독립 검증 충족까지 보장하지는 않습니다. 원문 대조와 출처 신뢰성의 수동 검토가 필요합니다. 실제 교수 평가 점수를 확정하거나 100점을 보장하지 않습니다.

## Contributors

공유 통합 코드를 역할별로 검토·반영한 내용을 포함하며, 모든 코드를 각자 처음 작성했다는 뜻은 아닙니다.

| 팀원 | 수행 역할 |
| --- | --- |
| 곽민규 | 이해관계자 평가, Agent JSON 응답 처리 |
| 김선정 | 종합·보고서 생성, 숫자 인용·출처 신뢰성 보완 |
| 이지원 | 시장성 평가, 근거·참고문헌 처리 |
| 임유리 | 기술 선정·조사, 실행 설정 및 중립성·인용 보완 |
| 현용찬 | 도메인·검색 평가, Graph·State·실행 진입점, 통합 검증·LangSmith Trace |

## 제출

GitHub 브랜치 링크 + 실제 LangSmith PNG + 최대 10장 평가 보고서 PDF를 함께 제출합니다.
압축 파일명: `Agent_판교캠퍼스_7반_곽민규+김선정+이지원+임유리+현용찬.zip`.
`dev` 병합 여부와 Slack 제출은 별도 승인 후 진행합니다.
