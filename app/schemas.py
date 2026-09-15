from pydantic import BaseModel


class SearchResult(BaseModel):
    id: int
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchResult]