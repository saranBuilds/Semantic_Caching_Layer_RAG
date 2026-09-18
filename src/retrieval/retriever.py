"""Ties the embedder and vector store together into one retrieve() call"""

from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore


class Retriever:
    def __init__(self,embedder:Embedder,store: VectorStore,top_k:int=4):
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self,query:str,query_embedding=None):
        if query_embedding is None:
            query_embedding = self.embedder.embed(query)
        return self.store.search(query_embedding,top_k = self.top_k)

