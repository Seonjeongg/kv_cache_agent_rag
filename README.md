# Subject

본 프로젝트는 KV cache 최적화 기술을 소프트웨어, 하드웨어 두 진영에서 선정하여
시장성·이해관계자·도메인 관점에서 중립적으로 비교 평가하는 Agentic RAG 시스템을
설계·개발한 프로젝트임.

## Overview

- Objective : 데이터센터·클라우드 LLM 서빙에서 KV cache 병목을 해결하는 두 접근(SW/HW)을
  기술 성숙도(TRL), 시장성, 이해관계자, 도메인 적용성 네 관점에서 비교 평가
- Method : LangGraph 기반 Multi-Agent(Fan-out/Fan-in) + Agentic RAG
- Tools : LangGraph, ChromaDB, Ollama, PyMuPDF, Tavily/DDGS

## Selected Technologies

- SW : DeepSeek-V2의 MLA(Multi-head Latent Attention) — 저차원 잠재 표현으로 KV cache 자체를
  압축하는 모델 구조 기반 접근. 메모리 사용량을 모델 구조에서 줄이는 방식을 분석하기 위해 선정
- HW : ITME(Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories) —
  CXL-Hybrid 메모리로 저장 계층을 확장하는 인프라 기반 접근. 저장 위치·계층 확장을 통한
  해결 방식을 분석하기 위해 선정
- 선정 방식 : Human-based (Agent 자동 선정이 아닌 팀 직접 선정)

## Features

- PDF 원문(DeepSeek-V2, ITME 논문) 기반 근거 추출 — PyMuPDF 파싱, 문자 단위 청킹(최대 1,400자,
  150자 중복), ChromaDB 로컬 벡터 저장소
- Tavily → DDGS → DuckDuckGo HTML 순서의 외부 웹 검색 대체 경로 (시장성·이해관계자 평가용)
- 모든 주요 주장에 evidence_id 연결, 공개 정보 부족 시 "공개 정보 부족"으로 명시
- 확증편향 방지 전략 : 서로 다른 실험 환경의 수치를 직접 우열 비교하지 않음, 기업 홍보 자료와
  독립 검증 결과 구분, 관련 생태계 자료를 해당 기술의 직접 근거로 오인하지 않도록 구분
- 약어 환각 방지 : 소형 로컬 모델이 MLA/ITME/CXL 등 약어를 임의로 잘못 풀어쓰는 것을 막기 위한
  고정 용어집(TERM_GLOSSARY)을 프롬프트에 주입

## Tech Stack

- Framework : LangGraph
- LLM/Generator : Ollama `qwen3:4b`
- Judge : Python 규칙 기반 검증 함수 (`validation_judge`, LLM 미사용 — 필수 분석 결과 존재 여부,
  evidence_id 등록 여부, 근거 목록 존재 여부를 코드로 검사)
- Retrieval : ChromaDB (Dense Retrieval, 코사인 거리) — Hit@K/MRR은 `evaluate.py`의 `EVAL_SET`에
  팀원이 원문을 읽고 직접 지정한 정답 청크 ID를 입력해야 측정됨 (현재 미입력 상태)
- Embedding : Qwen3-Embedding-0.6B (오픈소스, Apache 2.0) — 선정 사유는 설계서 2.6절 참고

## Agents

- 기술 조사 Agent : 원리·성능·한계·TRL 분석 (논문 RAG)
- 시장성 평가 Agent : 채택·생태계·도입 장벽·경제성 조사 (외부 웹 검색)
- 이해관계자 평가 Agent : 관계자별 기대·우려 조사 (외부 웹 검색)
- 도메인 평가 Agent : 데이터센터·클라우드 적용성 평가 (논문 RAG)
- 종합 평가 Agent : 관점 간 일치·상충·보완 가능성 정리
- 보고서 생성 Agent : 정해진 목차로 최종 보고서 작성
- 그 외 : 초기화 노드(technology_selection, Human이 선정한 기술 확정), 검증 노드(validation)

## Architecture

```
__start__ -> technology_selection -> technical_research
technical_research -> market_evaluation, stakeholder_evaluation, domain_evaluation (Fan-out)
[market_evaluation, stakeholder_evaluation, domain_evaluation] -> synthesis (Fan-in)
synthesis -> validation
validation --retry--> technical_research
validation --report--> report_generation -> __end__
```

전체 mermaid 정의는 설계서 4.4절 참고.

## Directory Structure

```
├── data/                    # 문서 풀 (논문 PDF)
├── agents/                  # Agent 모음 (역할별 분리)
├── config.py                # 환경변수, 경로, 모델·FAST_MODE 상수
├── state.py                 # Evidence, AgentState
├── rag.py                   # 논문 다운로드·파싱·청킹·임베딩·검색
├── llm.py                   # Ollama 호출 Helper
├── search.py                # 외부 웹 검색
├── evidence.py               # Evidence 생성·요약·참고문헌 포맷
├── graph.py                  # LangGraph 구성
├── app.py                    # 실행 스크립트
├── evaluate.py                # Retriever 평가 (Hit@K, MRR)
├── outputs/                   # 평가 결과 저장
└── tests/                     # 최소 self-check
```

프롬프트는 별도 `prompts/` 디렉토리 대신 각 `agents/*.py` 파일 내부에 함께 관리함
(Agent 로직과 프롬프트를 한 파일에서 같이 보는 편이 유지보수에 더 낫다고 판단).

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env   # TAVILY_API_KEY 입력 (없으면 DuckDuckGo로 자동 대체)
# Ollama가 로컬에서 실행 중이고 qwen3:4b / qwen3-embedding:0.6b가 설치되어 있어야 함
python app.py
```

`config.py`의 `FAST_MODE`를 `False`로 바꾸면 최종 실행 모드(재시도 최대 2회, Agent당 검색 5개,
기준별 질의)로 동작함. 제출용 최종 보고서는 `FAST_MODE = False`로 생성할 것.

## Contributors

<!-- TODO: PM·PL 역할 제외, 팀원별 실제 수행 역할로 교체 -->
- 곽민규 :
- 김선정 :
- 이지원 :
- 임유리 :
- 현용찬 :
