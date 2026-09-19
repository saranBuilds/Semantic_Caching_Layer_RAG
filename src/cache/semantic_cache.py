"""
In-memory semantic cache. Stores every question's embedding + answer,
and checks new questions against all of them using cosine similarity.
"""
import time
import numpy as np


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.entries = []  # each: {"embedding", "query", "answer", "added_at"}

    def get(self, query_embedding: np.ndarray):
        """
        Returns (cached_entry_or_None, best_similarity_score).
        best_similarity_score is 0.0 if the cache is empty, otherwise
        it's returned EVEN ON A MISS so we can see how close we got.
        """
        if not self.entries:
            return None, 0.0

        best_score = -1.0
        best_entry = None
        for entry in self.entries:
            # dot product of two normalized vectors == cosine similarity
            score = float(np.dot(query_embedding[0], entry["embedding"][0]))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.similarity_threshold:
            return {**best_entry, "similarity": best_score}, best_score
        return None, best_score

    def add(self, query_embedding: np.ndarray, query: str, answer: str):
        self.entries.append({
            "embedding": query_embedding,
            "query": query,
            "answer": answer,
            "added_at": time.time(),
        })