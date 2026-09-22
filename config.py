"""공통 설정: 환경변수, 경로, 모델·모드 상수, 클라이언트."""
from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(override=True)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY를 .env에 설정하세요.")

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_kv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

for directory in [RAW_DIR, CHROMA_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
COLLECTION_NAME = "kv_cache_papers"
TOP_K = 5

# 첫 실행은 빠른 구조 검증 모드로 진행합니다.
# 최종 보고서 생성 시에만 False로 바꾸세요.
FAST_MODE = True
MAX_RETRIES = 0 if FAST_MODE else 2
AGENT_RAG_TOP_K = 2 if FAST_MODE else 5
WEB_MAX_RESULTS = 1 if FAST_MODE else 4
FETCH_WEB_FULL_TEXT = not FAST_MODE
LLM_NUM_CTX = 8192 if FAST_MODE else 16384
JSON_NUM_PREDICT = 1400 if FAST_MODE else 2200
REPORT_NUM_PREDICT = 2400 if FAST_MODE else 7000

openai_client = OpenAI(api_key=OPENAI_API_KEY)

PAPERS = {
    "DeepSeek-V2 MLA": {
        "url": "https://arxiv.org/pdf/2405.04434",
        "path": RAW_DIR / "deepseek_v2.pdf",
        # arxiv.org/abs/2405.04434 에서 직접 확인한 서지정보 (REFERENCE 표기 형식용)
        "authors": "DeepSeek-AI",
        "year": "2024",
        "arxiv_id": "2405.04434",
        "title": "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model",
    },
    "ITME": {
        "url": "https://arxiv.org/pdf/2606.12556",
        "path": RAW_DIR / "itme.pdf",
        # arxiv.org/abs/2606.12556 에서 직접 확인한 서지정보 (REFERENCE 표기 형식용)
        "authors": "Jang, H., Min, Y., Kim, S., Ahn, T., Kim, H., Joo, Y., Kim, H., & Kim, J.",
        "year": "2026",
        "arxiv_id": "2606.12556",
        "title": "ITME: Inference Tiered Memory Expansion with Disaggregated CXL-Hybrid Memories",
    },
}
