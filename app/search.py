import csv
import pickle

import faiss
import numpy as np

from app.embeddings import EmbeddingModel


class SemanticSearch:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.embedding_model = EmbeddingModel()

        texts = [
            f"{doc['title']}. {doc['text']}"
            for doc in documents
        ]

        embeddings = self.embedding_model.encode(texts)

        self.embeddings = np.asarray(embeddings, dtype="float32")

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    @classmethod
    def load(
        cls,
        index_path: str,
        documents_path: str,
    ):
        instance = cls.__new__(cls)

        instance.embedding_model = EmbeddingModel()
        instance.index = faiss.read_index(index_path)

        with open(documents_path, "rb") as file:
            instance.documents = pickle.load(file)

        return instance

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            document = self.documents[index].copy()
            document["score"] = float(score)
            results.append(document)

        return results


def load_documents(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        documents = []

        for row in reader:
            documents.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "text": row["text"],
                }
            )

        return documents