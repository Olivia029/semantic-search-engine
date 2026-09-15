from app.search import SemanticSearch


search_engine = SemanticSearch.load(
    "models/documents.index",
    "models/documents.pkl",
)

results = search_engine.search(
    "How to deploy Kubernetes models?",
    top_k=5,
)

for result in results:
    print(
        f"{result['score']:.4f} | "
        f"{result['title']}"
    )