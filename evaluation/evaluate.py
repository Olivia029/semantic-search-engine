import csv

from app.search import SemanticSearch
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


QUERIES_PATH = "data/queries.csv"
INDEX_PATH = "models/documents.index"
DOCUMENTS_PATH = "models/documents.pkl"


def load_queries(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        queries = []

        for row in reader:
            queries.append(
                {
                    "query": row["query"],
                    "relevant_id": int(row["relevant_id"]),
                }
            )

        return queries


def main():
    search_engine = SemanticSearch.load(
        INDEX_PATH,
        DOCUMENTS_PATH,
    )

    queries = load_queries(QUERIES_PATH)

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    for item in queries:
        results = search_engine.search(
            item["query"],
            top_k=10,
        )

        retrieved_ids = [
            result["id"]
            for result in results
        ]

        relevant_ids = {item["relevant_id"]}

        precision = precision_at_k(
            retrieved_ids,
            relevant_ids,
            k=5,
        )

        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k=10,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_ranks.append(rr)

        print(
            f"{item['query']:<30} "
            f"P@5={precision:.2f} "
            f"R@10={recall:.2f} "
            f"RR={rr:.2f}"
        )

    precision_mean = sum(precision_scores) / len(
        precision_scores
    )

    recall_mean = sum(recall_scores) / len(
        recall_scores
    )

    mrr = sum(reciprocal_ranks) / len(
        reciprocal_ranks
    )

    print("\nOverall metrics")
    print(f"Precision@5: {precision_mean:.3f}")
    print(f"Recall@10:    {recall_mean:.3f}")
    print(f"MRR:          {mrr:.3f}")


if __name__ == "__main__":
    main()