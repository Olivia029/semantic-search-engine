from app.search import SemanticSearch


def get_search_engine():
    return SemanticSearch.load(
        "models/documents.index",
        "models/documents.pkl",
    )


def test_search_returns_results():
    search_engine = get_search_engine()

    results = search_engine.search(
        "semantic embeddings",
        top_k=5,
    )

    assert len(results) == 5


def test_search_returns_expected_document():
    search_engine = get_search_engine()

    results = search_engine.search(
        "semantic embeddings",
        top_k=5,
    )

    assert results[0]["id"] == 7
    assert results[0]["title"] == "Sentence Transformers"


def test_search_results_have_scores():
    search_engine = get_search_engine()

    results = search_engine.search(
        "docker containers",
        top_k=5,
    )

    for result in results:
        assert "score" in result
        assert isinstance(result["score"], float)
