"""
Core RAG + cache pipeline, decoupled from any interface (CLI or API).
Both main.py and the FastAPI routes use this same class so behavior
(and logging) stays identical no matter how a question comes in.
"""
import json
import os
import time

from src.ingestion.loader import load_pdf
from src.chunking.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.prompts.prompt_templates import RAG_SYSTEM_PROMPT, build_rag_prompt
from src.llm.llm_client import OllamaClient
from src.cache.semantic_cache import SemanticCache
from src.utils.helpers import get_logger


class RagPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = get_logger(cfg["logging"]["log_file"])

        self.logger.info("Loading embedding model...")
        self.embedder = Embedder(cfg["embeddings"]["model_name"], device=cfg["embeddings"]["device"])

        self.logger.info("Building or loading vector index...")
        self.store = self._build_or_load_index()
        self.retriever = Retriever(self.embedder, self.store, top_k=cfg["retrieval"]["top_k"])

        self.llm = OllamaClient(
            base_url=cfg["llm"]["base_url"],
            model=cfg["llm"]["model"],
            temperature=cfg["llm"]["temperature"],
        )

        self.cache_enabled = cfg["cache"]["enabled"]
        self.cache = SemanticCache(similarity_threshold=cfg["cache"]["similarity_threshold"])

    def _build_or_load_index(self) -> VectorStore:
        chunks_cache = self.cfg["paper"]["chunks_cache"]
        index_cache = self.cfg["paper"]["index_cache"]

        if os.path.exists(chunks_cache) and os.path.exists(index_cache):
            with open(chunks_cache, "r") as f:
                chunks = json.load(f)
            store = VectorStore(dim=self.embedder.embed("dummy").shape[1])
            store.load(index_cache)
            store.chunks = chunks
            return store

        pdf_path = self.cfg["paper"]["pdf_path"]
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(
                f"Place the paper PDF at {pdf_path} "
                "(download from https://arxiv.org/abs/1706.03762)"
            )

        text = load_pdf(pdf_path)
        chunks = chunk_text(
            text,
            chunk_size=self.cfg["chunking"]["chunk_size"],
            chunk_overlap=self.cfg["chunking"]["chunk_overlap"],
        )
        embeddings = self.embedder.embed(chunks)

        store = VectorStore(dim=embeddings.shape[1])
        store.add(embeddings, chunks)

        os.makedirs(os.path.dirname(chunks_cache), exist_ok=True)
        with open(chunks_cache, "w") as f:
            json.dump(chunks, f)
        store.save(index_cache)

        return store

    def answer(self, query: str) -> dict:
        """
        Runs one question through cache -> retrieval -> generation.
        Returns a structured dict — this is what both the CLI and the
        API endpoint use, so behavior never diverges between them.
        """
        start = time.time()
        query_embedding = self.embedder.embed(query)

        best_score = 0.0
        if self.cache_enabled:
            cached, best_score = self.cache.get(query_embedding)
            if cached:
                total_latency = round(time.time() - start, 3)
                self.logger.info(
                    f"query={query!r} | CACHE_HIT | matched={cached['query']!r} "
                    f"| similarity={cached['similarity']:.3f} | total_latency={total_latency}s"
                )
                return {
                    "answer": cached["answer"],
                    "cache_hit": True,
                    "similarity": cached["similarity"],
                    "matched_query": cached["query"],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_seconds": total_latency,
                }

        retrieved = self.retriever.retrieve(query, query_embedding=query_embedding)
        prompt = build_rag_prompt(query, retrieved)
        result = self.llm.generate(RAG_SYSTEM_PROMPT, prompt)
        total_latency = round(time.time() - start, 3)

        if self.cache_enabled:
            self.cache.add(query_embedding, query, result["answer"])

        self.logger.info(
            f"query={query!r} | CACHE_MISS | best_similarity={best_score:.3f} "
            f"| prompt_tokens={result['prompt_tokens']} | completion_tokens={result['completion_tokens']} "
            f"| llm_latency={result['latency_seconds']}s | total_latency={total_latency}s"
        )

        return {
            "answer": result["answer"],
            "cache_hit": False,
            "similarity": best_score,
            "matched_query": None,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "latency_seconds": total_latency,
        }