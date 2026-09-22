"""Retriever 평가: Hit@K, MRR. (design doc 10절)"""
from __future__ import annotations

from rag import retrieve

EVAL_SET = [
    # 예시 - 실제 정답 청크 ID를 확인한 뒤 주석을 해제하세요.
    # {
    #     "question": "How does MLA reduce KV cache?",
    #     "technology": "DeepSeek-V2 MLA",
    #     "ground_truth_chunk_id": "실제_chunk_id",
    # },
]


def evaluate_retriever(eval_set: list[dict], k: int = 5) -> dict:
    if not eval_set:
        return {"message": "EVAL_SET에 검증된 정답 청크를 입력하세요."}

    hits = 0
    reciprocal_ranks = []

    for item in eval_set:
        results = retrieve(item["question"], item["technology"], top_k=k)
        ids = [result["chunk_id"] for result in results]
        ground_truth = item["ground_truth_chunk_id"]

        if ground_truth in ids:
            rank = ids.index(ground_truth) + 1
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return {
        f"Hit@{k}": hits / len(eval_set),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "questions": len(eval_set),
    }


if __name__ == "__main__":
    print(evaluate_retriever(EVAL_SET, k=5))
