"""
In memory semantic cache .
stores every questions embedding+answer, 
and check new question against all of them using cosine similarity
"""

import time 
import numpy as np 


class SemanticCache:
    def __init__(self,similarity_thresold:float = 0.95):
        self.similarity_thresold = similarity_thresold
        self.entries = [] # each: {"embedding","query","answer","added_at"}

    def get(self,query_embedding:np.ndarray):
        """
        query_embedding: shape(1,dim) ,already normalized.
        return the best matching cached entry if similarity clears the thresold,
        else None.
        """
        if not self.entries:
            return None
        best_score = -1.0
        best_entry = None
        for entry in self.entries:
            #dot product of two normalized vectors == cosine similarity
            score = float(np.dot(query_embedding[0],entry["embeddings"][0]))
            if score>best_score:
                best_score = score
                best_entry = entry
        if best_score >= self.similarity_thresold:
            return {**best_entry,"similarity":best_score}
        return None

    def add(self,query_embedding:np.ndarray,query:str,answer:str):
        self.entries.append({
            "embedding" : query_embedding,
            "query" : query,
            "answer": answer,
            "added_at":time.time(),
        })