"""
Stub for Phase 2 of the caching-layer project.
Not wired into main.py yet — main.py runs as a CLI for now.
This will later expose a /v1/chat/completions-style endpoint that
checks the semantic cache before calling the RAG pipeline below.
"""
from fastapi import FastAPI

app = FastAPI(title="RAG Cache Proxy (Phase 2 placeholder)")


@app.get("/health")
def health():
    return {"status": "ok"}

# TODO (Phase 2): add POST /ask that runs retrieval + generation
# TODO (Phase 1): check semantic cache before hitting the pipeline