# 최신 dev 평가 기준 점검

## 요약 판정

- 점검 대상: 팀 저장소 dev `a8fa028afb0aa777dad5d681c8ce8256f952940e`.
- 개인 작업 저장: `feature/hyc-validation-trace`, 기능 커밋 `47dfe21`.
- dev는 수정하거나 병합하지 않았다. 임시 복사본에서 테스트와 진단을 실행했다.
- 팀원의 숫자 인용, citation_map, 출처 분류, 시장/이해관계자 프롬프트는 유지하는 방향이 적절하다.
- 개인 브랜치 전체 병합은 추천하지 않는다. merge-tree 검사에서 README.md, agents/synthesis.py, app.py, config.py, graph.py, tests/test_smoke.py의 6개 파일에 충돌이 확인됐다. 실제 merge는 하지 않았다.
- 최신 dev는 코드 구조가 상당 부분 갖춰졌지만, 최신 코드로 생성한 최종 보고서와 실제 LangSmith 캡처를 확인하지 못했으므로 제출 완료나 100점 충족으로 판정하지 않는다.

## 데이터 신뢰도

| 구분 | 확인한 내용 |
| --- | --- |
| 직접 측정 | dev의 `python tests/test_smoke.py`: 34개 통과 |
| 직접 측정 | 개인 브랜치의 동일 테스트: 31개 통과 |
| 직접 측정 | 요청하지 않은 관점 재실행, 미등록 인용 제거 후 주장 유지, 중립 문장 오탐, 독립 출처 없이 source_reliability 통과 사례 |
| 코드 확인 | 계획, Send, reducer, fallback, 품질 재작업, 종료 상한, 숫자 인용 역매핑 |
| 확인하지 않음 | 최신 dev 전체 API 실행, 최신 dev 보고서의 사실 정확성/페이지 수, 최신 dev 실제 LangSmith 화면 |
| 비교 불가 | 개발 시간, 토큰 효율, 생산성 개선율. 동일 조건의 비교 자료 없음 |

개인 브랜치의 4장 보고서는 `warning / pass_with_limitations`였고 중립성 항목이 실패했다. 이는 최신 dev 실행 결과가 아니며 dev의 품질을 대신 입증하지 않는다.

## 교수님 평가표 대응

| 항목 | 배점 | 판정 | 근거와 남은 확인 |
| --- | ---: | --- | --- |
| 패턴 적용 정합성 | 20 | 대부분 구현, 일부 보완 | State plan -> Send -> Worker -> Synthesizer. 실패 기록 후 계속 진행, 재시도/종료 상한 있음. 부분 관점 요청의 validation은 불필요한 관점을 추가 실행함 |
| 동적 동작 실증 | 20 | 코드/단위 테스트 확인, 제출 증빙 미확인 | 요청에 따라 1~4개 Send 생성. 저장소의 local_trace는 실제 LangSmith 제출 캡처가 아님. 최신 dev의 trace가 필요 |
| State Schema 설계 | 20 | 대부분 구현, 설명/연결 일부 보완 | 제어/페이로드, reducer, 상태/오류/재시도/종료 필드 있음. trace_id의 명시적 root run 연결 없음. checkpointer 없음. README의 식별자만 저장한다는 설명과 evidence_text 저장이 다름 |
| 품질 평가 노드 | 15 | 구현 확인, 동작 위험 있음 | 보고서 생성 뒤 4개 최소 항목 및 source_reliability 검사, worker/report 재시도. 중립성 오탐, 미등록 근거 주장 잔존, 독립 출처 판정의 한계가 있음 |
| 코드 구조 및 모듈 분리 | 5 | 구현 충족, 문서 보완 | 조정 계층과 agents 분리. 프롬프트를 Agent 파일 안에 두는 이유 명시. Architecture 이미지가 구 버전이라 실제 Graph와 불일치 |
| 실행 결과 재현성 | 10 | 단위 테스트 확인, 전체 실행 미확인 | 34개 테스트 통과. 최신 dev 전체 보고서 생성 및 동일 실행 trace 대조는 아직 확인하지 않음 |
| Output - 보고서 | 10 | 생성/검증 코드 있음, 산출물 미확인 | SUMMARY/REFERENCE 및 숫자 인용 검증, PDF 10장 제한. 저장소 baseline은 이전 RAG 결과로 설명되어 있음. 최신 dev 생성물로 검증 필요 |

배점은 교수님 기준이다. 실제 채점 결과나 확정 예상 점수는 아니다. 가이드의 1안은 형식 중심의 결정론적 평가를 허용하므로 LLM Judge 미사용 자체는 감점 사유로 단정하지 않는다.

## 확인된 문제와 의미

### 1. 부분 관점 요청에 없는 Worker까지 재실행

`agents/synthesis.py:validation_judge`는 required_task_ids를 사용하지 않고 네 관점 결과를 전부 요구한다. 기술/도메인만 있는 정상 State를 검사했을 때 시장/이해관계자 누락으로 retry가 반환됐다. 처음 fan-out이 동적이라는 점은 유효하지만, 전체 실행 경로의 부분 관점 유지에는 문제가 있다.

추가 추천: 팀원 숫자 인용 변환을 유지하고, 검증 대상만 최초 required_task_ids에 한정하는 작은 보완. 별도 회귀 테스트 추가.

### 2. 미등록 근거는 사라져도 주장은 남음

`exclude_unregistered_evidence("성능이 99% 향상됐다. [rag-bbbbbbbbbbbb]", {"rag-aaaaaaaaaaaa"})`를 실행하면 성능 주장은 남고 추가 검증 필요 안내가 붙는다. 문장 뒤의 인용이 분리되면 안내도 주장과 분리될 수 있다.

형식 검사가 사실 검증을 대신하지 않으며, 근거 없는 단정 문장의 노출 위험이 있다. 개인 브랜치의 문장 단위 제외 정책을 별도 후처리 함수로 추가하는 방법을 추천한다. 숫자 인용 변환은 유지한다.

### 3. 중립성 검사의 오탐

`승자를 정하는 것이 아니라 조건 차이를 비교한다.`가 기존 정규식의 `승자`에 걸린다. 검사가 REFERENCE를 포함한 전체 보고서를 대상으로 하므로 출처 제목도 본문 판정에 영향을 줄 수 있다.

추가 추천: 본문 판정 범위를 분리하고 부정 문장 테스트를 추가. 이는 단순 문서 추가만으로 해결되지 않는 작은 검사 로직 보완이다. 개인 브랜치의 금지어 확대를 통째로 복사하면 오탐이 늘 수 있으므로 그대로 적용하지 않는다.

### 4. 독립 출처를 기록하지만 충분성을 강제하지 않음

INDEPENDENT_DOMAINS는 빈 집합이다. github.com과 deepseek.com의 공식 출처 두 개만 사용한 시장성 fixture에서도 independent_source_count=0, source_reliability.passed=true였다. 즉 서로 다른 도메인 검사와 독립 검증은 같은 의미가 아니다.

이는 교수님이 출처 등급 1~5 체계를 의무화했다는 뜻은 아니다. 팀이 선언한 독립 검증 정책과 실제 검사 사이의 차이다. 원 논문이라는 이유만으로 해당 기술의 독립 검증 자료가 되는 것도 아니다.

unclassified를 자동으로 저품질로 분류하지 않는 설계는 유지한다. 다만 출처 미분류 경고, 사실/공식 발표/해석 구분, 독립 검증 부족 표시를 별도 점검으로 추가하는 편이 적절하다. 실제 출처 내용을 확인하지 않고 임의로 도메인을 독립 출처 목록에 넣지 않는다.

### 5. 문서와 실행 증빙

- Architecture PNG는 technical_research 이후 고정 분기하는 구 흐름이며 orchestrator_plan/quality_evaluator가 없다.
- README에 내부 evidence_id 최종 출력 설명이 남아 숫자 인용 코드와 다르다.
- Contributors는 기존 RAG 커밋 중심이며 이번 과제의 구체적 수행 파일/역할은 별도 보완이 유용하다. PM/PL 역할은 요구되지 않는다.
- references에 evidence_text가 저장되므로 State가 식별자만 저장한다는 설명은 부정확하다. 반복/검색 상한으로 누적 비용을 제한한다는 설명이 맞다.
- build_graph().checkpointer는 None이다. 한 실행 안의 재시도와 프로세스 중단 후 재개를 구분해 설명해야 한다. checkpointer 자체를 교수님이 반드시 요구한 것으로 해석하지 않는다.
- LangGraph는 환경변수로 tracing할 수 있으나 현재 app.py는 State trace_id를 명시적 run_id/metadata로 연결하지 않고, OpenAI SDK 호출도 wrap_openai로 감싸지 않았다.

## 개인 작업 선별 추가안

| 우선순위 | 추가할 내용 | 팀원 코드 보존 방법 |
| --- | --- | --- |
| 1 | tracing.py, 실행 root ID/metadata/callback, OpenAI 호출 추적 | 신규 모듈 + config/app 실행 연결 부분만 추가. 출처 목록 및 숫자 인용 검증은 유지 |
| 1 | 최신 Graph 이미지와 State 설명 보충 문서 | 기존 구현은 그대로 두고 새 문서/이미지 추가 및 README에 연결 |
| 1 | 별도 통합/guardrail 테스트 | tests/test_integration_guards.py 등 신규 파일. 팀원 34개 테스트 유지 |
| 2 | demo_fanout.py | 별도 증빙 실행 파일. 보고서 전 의도적 중단을 명시하며 제출 보고서로 혼동하지 않음 |
| 2 | retrieval_metrics 저장, 기존 index 재사용 옵션 | 검색 알고리즘을 바꾸지 않고 평가 결과/실행 옵션 추가. 개인 측정 Hit@5=0.50, MRR=0.3833은 dev 새 측정으로 오인하지 않음 |
| 2 | 보고서 CSS 연결 및 페이지 검사 자원 정리 | 숫자 인용 검증을 보존하고 렌더링 연결만 보완 |
| 별도 합의 | 부분 관점 validation, 미등록 주장 제외, 중립성 검사 보완 | 사실 확인된 최소 범위만 보완. 전체 synthesis/graph 교체 금지 |

기존 코드를 한 줄도 수정하지 않으면 문서/별도 테스트/별도 실행 파일은 추가할 수 있지만, 기존 app 실행 경로에 적용되는 로깅 연결과 확인된 버그 수정까지 완성할 수는 없다. 추가 중심 방식도 최소 연결 지점 수정이 필요한 부분은 명확히 구분해야 한다.

## Ollama 사용 이유

- 현재 OpenAI gpt-4o-mini가 분석/보고서를 생성하고 Ollama qwen3-embedding:0.6b가 논문과 검색 질의를 벡터로 바꾼다.
- rag.py:embed_texts와 retrieve의 질의 임베딩이 Ollama를 직접 호출한다. 현재 코드를 그대로 실행하려면 Ollama가 필요하다.
- 과제는 오픈소스 임베딩 모델을 기록하도록 했지만 Ollama라는 실행 도구를 의무화하지 않았다. 다른 로컬 실행 방식으로 대체할 수 있다.
- 장점은 임베딩 API 호출 과금 없이 로컬에서 문서/질의를 처리하는 것. 대신 설치, 모델 다운로드, 로컬 연산 시간/자원은 필요하다.
- 교체하려면 문서/질의 두 임베딩 경로, 의존성, 실행 문서, 색인을 함께 검증해야 한다. 모델/벡터 설정이 달라지면 기존 색인을 재생성하고 Hit@K/MRR을 다시 측정한다.
- 임베딩만 로컬이라는 뜻이며 전체 서비스가 오프라인/외부 전송 없음이라는 뜻은 아니다. 분석은 OpenAI, 웹 검색은 외부 서비스에 요청한다.

공식 문서: [Ollama embeddings](https://docs.ollama.com/capabilities/embeddings), [Sentence Transformers quickstart](https://sbert.net/docs/quickstart.html).

## 다음 작업 규칙

최신 dev 기반 통합용 브랜치에서 추적/문서/별도 테스트만 선별 추가하고 기존 34개 테스트와 새 사례를 함께 검사한다. 이후 FAST_MODE=false 전체 실행, 숫자 인용/출처 연결, 품질 결과, 실제 LangSmith 캡처, 10장 이하 PDF를 같은 run 기준으로 확인한다. dev 병합은 사용자의 별도 승인 이후에만 진행한다.
