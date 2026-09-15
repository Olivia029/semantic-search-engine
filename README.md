# Semantic Search Engine

A semantic search engine for finding relevant documents based on meaning rather than exact keyword matches.

The project uses Sentence Transformers to generate text embeddings and FAISS to perform vector similarity search. A FastAPI service exposes the search engine through a REST API, while a Streamlit interface provides a simple web UI.

## Architecture

```text
User query
    │
    ▼
MiniLM Embedding Model
    │
    ▼
384-dimensional vector
    │
    ▼
FAISS similarity search
    │
    ▼
Top-K documents
    │
    ▼
FastAPI / Streamlit
```

The document collection is embedded once when the FAISS index is built. At query time, only the search query needs to be encoded before the nearest vectors are retrieved.

## Tech Stack

* Python 3.12
* Sentence Transformers
* `all-MiniLM-L6-v2`
* FAISS
* FastAPI
* Pydantic
* Streamlit
* Pandas
* NumPy
* scikit-learn
* Pytest
* Docker
* Docker Compose

## Project Structure

```text
semantic-search-engine/
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── embeddings.py
│   ├── schemas.py
│   └── search.py
├── data/
│   ├── documents.csv
│   └── queries.csv
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   └── metrics.py
├── models/
│   ├── documents.index
│   └── documents.pkl
├── scripts/
│   ├── __init__.py
│   ├── build_index.py
│   └── test_search.py
├── tests/
│   ├── __init__.py
│   └── test_search.py
├── ui/
│   └── streamlit_app.py
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## How Search Works

Documents are represented using their title and text:

```text
Document = title + text
```

The `all-MiniLM-L6-v2` model converts each document into a 384-dimensional embedding.

The embeddings are normalized and stored in a FAISS `IndexFlatIP` index. Inner product is then used to compare the query vector with the document vectors.

For a query such as:

```text
semantic embeddings
```

the search engine returns the documents with the highest similarity scores.

Example:

```text
0.6281 | Sentence Transformers
0.5797 | Vector Databases
0.2137 | PyTorch Training
0.0973 | Machine Learning Metrics
0.0963 | Streamlit Dashboard
```

## Dataset

The current dataset contains 10 technical documents covering topics such as:

* Kubernetes
* Docker
* FastAPI
* PyTorch
* Git
* FAISS
* Sentence Transformers
* CI/CD
* Machine Learning metrics
* Streamlit

The evaluation set contains six queries, each associated with one relevant document.

The dataset is intentionally small. It is mainly used to validate the complete search pipeline and provide a reproducible example.

## Running Locally

Clone the repository:

```bash
git clone https://github.com/Olivia029/semantic-search-engine.git
cd semantic-search-engine
```

Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Build the FAISS index:

```bash
python -m scripts.build_index
```

This creates:

```text
models/documents.index
models/documents.pkl
```

## Running the API

Start FastAPI with Uvicorn:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

The API will be available on:

```text
http://localhost:8000
```

The root endpoint can be used to check whether the service is running:

```bash
curl http://localhost:8000/
```

Example response:

```json
{
  "message": "Semantic Search Engine API",
  "status": "running"
}
```

### Search Endpoint

Search for documents with:

```bash
curl "http://localhost:8000/search?q=semantic%20embeddings"
```

The endpoint accepts an optional `top_k` parameter:

```bash
curl "http://localhost:8000/search?q=docker%20containers&top_k=5"
```

The response contains the original query, the number of returned documents, document information and the similarity score for each result.

## Streamlit Interface

The project also includes a Streamlit interface.

Start the API first, then run:

```bash
streamlit run ui/streamlit_app.py
```

The interface allows the user to:

* Enter a natural-language query
* Choose the number of results
* Run a semantic search
* View the returned documents
* Compare similarity scores

The Streamlit application communicates with the FastAPI service rather than accessing the search engine directly.

## Docker

The API can be built and started with Docker Compose:

```bash
docker compose up -d --build
```

Check the running service:

```bash
docker compose ps
```

Expected output:

```text
NAME           SERVICE   STATUS
semantic-api   api       Up
```

The API is exposed on port `8000`.

Test it with:

```bash
curl "http://localhost:8000/search?q=semantic%20embeddings"
```

Stop the service with:

```bash
docker compose down
```

## Evaluation

The search engine includes a small evaluation pipeline using three ranking metrics:

### Precision@5

Measures how many of the first five retrieved documents are relevant.

### Recall@10

Measures whether the relevant document appears within the first ten results.

### Mean Reciprocal Rank

Measures how high the first relevant result appears in the ranking.

Run the evaluation with:

```bash
python -m evaluation.evaluate
```

Current results on the included evaluation set:

```text
Overall metrics
Precision@5: 0.200
Recall@10:    1.000
MRR:          1.000
```

The dataset contains one relevant document per query. As a result, a relevant document in the first five positions gives a Precision@5 of `0.20`, while finding it anywhere in the first ten positions gives a Recall@10 of `1.00`.

The MRR of `1.00` means that the relevant document is ranked first for every query in the current evaluation set.

These numbers should not be interpreted as a benchmark for a production search system. The evaluation dataset is deliberately small.

## Tests

The project includes Pytest tests covering the main search behavior.

Run:

```bash
pytest
```

The current test suite checks that:

* Searches return the requested number of results
* Relevant documents are ranked correctly for selected queries
* Search results contain numerical similarity scores

## Current Limitations

The current implementation is intentionally compact.

Some areas that could be improved include:

* A larger and more representative document collection
* Multiple relevant documents per query
* A larger evaluation set
* More robust ranking evaluation
* Metadata filtering
* Incremental index updates
* A production-ready vector database
* Authentication and API rate limiting
* Better error handling
* Container health checks
* CI/CD automation

## Possible Next Steps

A natural progression for the project would be:

1. Expand the evaluation dataset.
2. Add more relevant documents per query.
3. Compare different embedding models.
4. Experiment with ranking and retrieval strategies.
5. Add automated tests to CI.
6. Improve the Streamlit interface.
7. Deploy the API and UI publicly.

## License

This project is provided for educational and portfolio purposes.

---

*This README.md was formatted with AI assistance using ChatGPT.*
