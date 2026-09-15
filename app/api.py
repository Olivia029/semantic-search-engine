from app.schemas import SearchResponse

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=10)
):
    results = search_engine.search(q, top_k)

    return SearchResponse(
        query=q,
        total_results=len(results),
        results=results
    )