"""Retriever 평가: Hit@K, MRR 및 질문별 검색 결과 상세 정보."""
from __future__ import annotations

import json
from typing import Any

from rag import build_index, download_papers, load_and_chunk_papers, retrieve

# 현용찬: 초기 평가셋 예시를 실제 MLA·ITME 질문 10개와 정답 청크 ID로
# 구체화했다. 페이지 번호가 아니라 청크를 정답으로 삼아 현재 RAG
# 파이프라인의 검색 단위를 직접 검증한다.
EVAL_SET = [
    {
        "id": "mla-core-mechanism",
        "question": "How does DeepSeek-V2 MLA represent keys and values to reduce KV cache?",
        "technology": "DeepSeek-V2 MLA",
        "ground_truth_chunk_ids": ["a9b797947a55c2abffb5", "a0c0c90446dbc74b4f11"],
    },
    {
        "id": "mla-memory-reduction",
        "question": "What KV cache reduction is reported for DeepSeek-V2 MLA and under what comparison?",
        "technology": "DeepSeek-V2 MLA",
        "ground_truth_chunk_ids": ["eee163f1c91fd182fd1f", "430a45f2bcbca3af0d5e"],
    },
    {
        "id": "mla-limitations",
        "question": "What deployment or architecture constraints does DeepSeek-V2 MLA have?",
        "technology": "DeepSeek-V2 MLA",
        "ground_truth_chunk_ids": ["0f3bcc79fea662df66d5", "860fef559d533ffddc42"],
    },
    {
        "id": "mla-long-context",
        "question": "What does the DeepSeek-V2 paper report about long-context efficiency or context length?",
        "technology": "DeepSeek-V2 MLA",
        "ground_truth_chunk_ids": ["a6d0dec26c6f3b97981a", "793419bcadda5a8e1753"],
    },
    {
        "id": "mla-readiness",
        "question": "What evidence in DeepSeek-V2 indicates system-scale evaluation or technology readiness?",
        "technology": "DeepSeek-V2 MLA",
        "ground_truth_chunk_ids": ["82bc570c6dce4e06ce58", "8ca8829261cd80970d61"],
    },
    {
        "id": "itme-core-mechanism",
        "question": "How does ITME use disaggregated CXL-Hybrid Memory for LLM inference?",
        "technology": "ITME",
        "ground_truth_chunk_ids": ["93a4b64e9a64ed1a9e42", "c628372184a2fb491e5a"],
    },
    {
        "id": "itme-tiering",
        "question": "What data placement or tiering strategy does ITME describe for KV cache?",
        "technology": "ITME",
        "ground_truth_chunk_ids": ["629461a39f68d526d8bf", "fc003c5cc1c3cb5fa6cf"],
    },
    {
        "id": "itme-latency",
        "question": "What latency or data-transfer trade-offs are reported for ITME?",
        "technology": "ITME",
        "ground_truth_chunk_ids": ["5c33b452f05c97f8f1c9", "c06f1ec3217b70253546"],
    },
    {
        "id": "itme-throughput",
        "question": "What throughput result and evaluation conditions are reported for ITME?",
        "technology": "ITME",
        "ground_truth_chunk_ids": ["70f14e2a229663480315", "1ed045a34e6cefa6cdce"],
    },
    {
        "id": "itme-readiness",
        "question": "What system prototype or evaluation evidence supports the readiness of ITME?",
        "technology": "ITME",
        "ground_truth_chunk_ids": ["b989221ece0c9603d8be", "22a2015fc92ce21c82d4"],
    },
]

EVAL_K = 5
CANDIDATE_K = 10
EvalCase = dict[str, Any]


def _expected_targets(item: EvalCase) -> tuple[set[str | int], str]:
    """평가 케이스에서 정답 집합과 검색 결과의 비교 필드를 반환한다."""
    if "ground_truth_chunk_ids" in item:
        expected = {str(chunk_id) for chunk_id in item["ground_truth_chunk_ids"]}
        if not expected:
            raise ValueError(f"정답 청크 ID가 비어 있습니다: {item.get('id')}")
        return expected, "chunk_id"

    if "ground_truth_chunk_id" in item:
        return {str(item["ground_truth_chunk_id"])}, "chunk_id"

    if "expected_pages" in item:
        return {int(page) for page in item["expected_pages"]}, "page"

    raise ValueError(
        f"{item.get('id', '<unknown>')}에 ground_truth_chunk_ids, "
        "ground_truth_chunk_id 또는 expected_pages가 필요합니다."
    )


def _evaluate_case(
    item: EvalCase,
    *,
    k: int,
    candidate_k: int,
    rerank: bool,
) -> dict[str, Any]:
    """한 질문을 평가하고 재현 가능한 상세 결과를 만든다."""
    expected, match_key = _expected_targets(item)

    # 현용찬: 먼저 후보를 넉넉히 검색한 뒤 lexical rerank를 적용한다.
    # 임베딩 검색의 후보 누락 여부와 최종 top-k 순위를 함께 확인하기 위한 단계다.
    results = retrieve(
        item["question"],
        item["technology"],
        top_k=k,
        candidate_k=max(candidate_k, k),
        rerank=rerank,
    )

    rank = None
    for index, result in enumerate(results, start=1):
        if result.get(match_key) in expected:
            rank = index
            break

    retrieved_chunk_ids = [
        result.get("chunk_id") for result in results if result.get("chunk_id")
    ]

    # 현용찬: 점수만 출력하지 않고 질문별 rank·매칭 청크·검색 후보를
    # 함께 저장해 실패한 질문을 사람이 재검토할 수 있도록 했다.
    return {
        "id": item.get("id"),
        "technology": item["technology"],
        "rank": rank,
        "matched_chunk_id": results[rank - 1].get("chunk_id") if rank else None,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "match_key": match_key,
        "reciprocal_rank": 1 / rank if rank else 0,
    }


def evaluate_retriever(
    eval_set: list[EvalCase],
    k: int = EVAL_K,
    candidate_k: int = CANDIDATE_K,
    rerank: bool = True,
) -> dict[str, Any]:
    """평가셋에 대해 Hit@K, MRR과 질문별 진단 결과를 계산한다."""
    if k <= 0:
        raise ValueError("k는 1 이상의 정수여야 합니다.")
    if candidate_k <= 0:
        raise ValueError("candidate_k는 1 이상의 정수여야 합니다.")

    if not eval_set:
        return {"message": "EVAL_SET에 검증된 정답을 입력하세요.", "evaluated": False}

    details = [
        _evaluate_case(
            item,
            k=k,
            candidate_k=candidate_k,
            rerank=rerank,
        )
        for item in eval_set
    ]
    hits = sum(detail["rank"] is not None for detail in details)
    reciprocal_ranks = [detail["reciprocal_rank"] for detail in details]

    return {
        f"Hit@{k}": hits / len(details),
        "MRR": sum(reciprocal_ranks) / len(details),
        "questions": len(details),
        "evaluated": True,
        "details": details,
    }


if __name__ == "__main__":
    # 현용찬: 실행할 때 논문 다운로드부터 색인 생성까지 다시 수행한다.
    # 임베딩 모델·청킹 설정이 바뀌어도 오래된 Chroma 색인을 재사용하지 않아
    # 평가 결과를 동일한 설정에서 재현할 수 있다.
    download_papers()
    chunks = load_and_chunk_papers()
    collection = build_index(chunks)
    print(f"평가용 색인 완료: 청크 {collection.count():,}개")
    print(
        json.dumps(
            evaluate_retriever(EVAL_SET),
            ensure_ascii=False,
            indent=2,
        )
    )
