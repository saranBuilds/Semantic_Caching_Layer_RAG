"""
Entry point: builds (or loads a cached) FAISS index over the paper,
then runs an interactive Q&A loop. A semantic cache sits in front of
retrieval+generation so repeated/similar questions skip the LLM call.
"""
import json
import os
import time
import yaml

from src.ingestion.loader import load_pdf
from src.chunking.chunker import chunk_text
from src.embeddings.embedder import Embedder
from src.vectordb.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.prompts.prompt_templates import RAG_SYSTEM_PROMPT, build_rag_prompt
from src.llm.llm_client import OllamaClient
from src.cache.semantic_cache import SemanticCache
from src.utils.helpers import get_logger


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_or_load_index(cfg: dict, embedder: Embedder) -> VectorStore:
    chunks_cache = cfg["paper"]["chunks_cache"]
    index_cache = cfg["paper"]["index_cache"]

    if os.path.exists(chunks_cache) and os.path.exists(index_cache):
        with open(chunks_cache, "r") as f:
            chunks = json.load(f)
        store = VectorStore(dim=embedder.embed("dummy").shape[1])
        store.load(index_cache)
        store.chunks = chunks
        return store

    pdf_path = cfg["paper"]["pdf_path"]
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Place the paper PDF at {pdf_path} "
            "(download from https://arxiv.org/abs/1706.03762)"
        )

    text = load_pdf(pdf_path)
    chunks = chunk_text(
        text,
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"],
    )
    embeddings = embedder.embed(chunks)

    store = VectorStore(dim=embeddings.shape[1])
    store.add(embeddings, chunks)

    os.makedirs(os.path.dirname(chunks_cache), exist_ok=True)
    with open(chunks_cache, "w") as f:
        json.dump(chunks, f)
    store.save(index_cache)

    return store


def main():
    cfg = load_config()
    logger = get_logger(cfg["logging"]["log_file"])

    logger.info("Loading embedding model...")
    embedder = Embedder(cfg["embeddings"]["model_name"], device=cfg["embeddings"]["device"])

    logger.info("Building or loading vector index...")
    store = build_or_load_index(cfg, embedder)
    retriever = Retriever(embedder, store, top_k=cfg["retrieval"]["top_k"])

    llm = OllamaClient(
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
        temperature=cfg["llm"]["temperature"],
    )

    cache_enabled = cfg["cache"]["enabled"]
    cache = SemanticCache(similarity_threshold=cfg["cache"]["similarity_threshold"])

    print("\nReady. Ask questions about 'Attention Is All You Need' (type 'exit' to quit).\n")

    while True:
        query = input("> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        start = time.time()
        query_embedding = embedder.embed(query)

        if cache_enabled:
            cached = cache.get(query_embedding)
            if cached:
                total_latency = round(time.time() - start, 3)
                print(f"\n[cache hit — similarity={cached['similarity']:.3f}]\n{cached['answer']}\n")
                logger.info(
                    f"query={query!r} | CACHE_HIT | matched={cached['query']!r} "
                    f"| similarity={cached['similarity']:.3f} | total_latency={total_latency}s"
                )
                continue

        retrieved = retriever.retrieve(query, query_embedding=query_embedding)
        prompt = build_rag_prompt(query, retrieved)
        result = llm.generate(RAG_SYSTEM_PROMPT, prompt)
        total_latency = round(time.time() - start, 3)

        print(f"\n{result['answer']}\n")

        if cache_enabled:
            cache.add(query_embedding, query, result["answer"])

        logger.info(
            f"query={query!r} | CACHE_MISS | prompt_tokens={result['prompt_tokens']} "
            f"| completion_tokens={result['completion_tokens']} "
            f"| llm_latency={result['latency_seconds']}s | total_latency={total_latency}s"
        )


if __name__ == "__main__":
    main()