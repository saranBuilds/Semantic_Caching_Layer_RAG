RAG_SYSTEM_PROMPT = (
    "you are a research assistant answering question ONLY about the paper"
    "'Attention Is ALL tou Need'. Use the provided context chunks."
    "If the context doesn't contain the answer, say so instead of guessing."
)

def build_rag_prompt(query:str,retrieved_chunks:list)->str:
    context = "\n\n".join(
        f"[Chunk {i+1}]\n{item['chunk']}" for i,item in enumerate(retrieved_chunks)
    )
    return(
        f"Context from the paper:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above:"
    )