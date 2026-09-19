"""Thin FAISS wrapper. In-memory is fine at single-paper scale."""
import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []  # parallel list: chunks[i] matches vector i

    def add(self, embeddings: np.ndarray, chunks: list):
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 4):
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"chunk": self.chunks[idx], "score": float(score)})
        return results

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)