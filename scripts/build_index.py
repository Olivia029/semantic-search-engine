import pickle

import faiss

from app.search import SemanticSearch, load_documents


DOCUMENTS_PATH = "data/documents.csv"
INDEX_PATH = "models/documents.index"
DOCUMENTS_INDEX_PATH = "models/documents.pkl"


def main():
    documents = load_documents(DOCUMENTS_PATH)

    search_engine = SemanticSearch(documents)

    faiss.write_index(
        search_engine.index,
        INDEX_PATH,
    )

    with open(DOCUMENTS_INDEX_PATH, "wb") as file:
        pickle.dump(documents, file)

    print(f"Indexed {len(documents)} documents")
    print(f"FAISS index saved to {INDEX_PATH}")
    print(f"Documents saved to {DOCUMENTS_INDEX_PATH}")


if __name__ == "__main__":
    main()