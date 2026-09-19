"""Wraps the embedding model so the rest of the app doesn't care which one it is."""
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts) -> np.ndarray:
        """Embed a string or list of strings. Always returns a 2D float32 array."""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")