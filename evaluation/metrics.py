def precision_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        doc_id in relevant_ids
        for doc_id in retrieved
    )

    return relevant_retrieved / k


def recall_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    retrieved = retrieved_ids[:k]

    if not relevant_ids:
        return 0.0

    relevant_retrieved = sum(
        doc_id in relevant_ids
        for doc_id in retrieved
    )

    return relevant_retrieved / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[int],
    relevant_ids: set[int],
) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0