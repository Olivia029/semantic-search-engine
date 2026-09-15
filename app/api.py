from fastapi import FastAPI, Query
from app.search import SemanticSearch

app = FastAPI(
    title="Semantic Search Engine",
    description="Semantic document search using MiniLM + FAISS",
    version="1.0.0"
)

# Cargar el índice UNA sola vez al iniciar la API
search_engine = SemanticSearch.load(
    "models/documents.index",
    "models/documents.pkl"
)


@app.get("/")
def home():
    return {
        "message": "Semantic Search Engine API",
        "status": "running"
    }


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=10)
):
    results = search_engine.search(q, top_k)

    return {
        "query": q,
        "results": results
    }