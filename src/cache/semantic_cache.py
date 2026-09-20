"""
In-memory semantic cache. Stores every question's embedding + answer,
and checks new questions against all of them using cosine similarity.
Can persist to disk so answers survive a restart.
"""
import os
import pickle
import time
import numpy as np


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.entries = []  # each: {"embedding", "query", "answer", "added_at"}

    def get(self, query_embedding: np.ndarray):
        """
        Returns (cached_entry_or_None, best_similarity_score).
        best_similarity_score is returned even on a miss, for debugging.
        """
        if not self.entries:
            return None, 0.0

        best_score = -1.0
        best_entry = None
        for entry in self.entries:
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

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.entries, f)

    def load(self, path: str):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.entries = pickle.load(f)