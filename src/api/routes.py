""""""


from fastapi import FastAPI

app = FastAPI(title="RAG Cache Proxy (Phase 2 placeholder)")

@app.get("/health")
def health():
    return {"status":"ok"}