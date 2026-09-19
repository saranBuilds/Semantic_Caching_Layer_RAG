"""
Phase 2: drop-in API front door. Loads the pipeline ONCE at startup,
then exposes it over HTTP so any client (curl, Postman, a future
frontend) can ask questions without touching Python.
"""
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline import RagPipeline

app = FastAPI(title="RAG Cache Proxy — Attention Is All You Need")

pipeline: RagPipeline | None = None


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def load_pipeline():
    global pipeline
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    pipeline = RagPipeline(cfg)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    return pipeline.answer(request.question)