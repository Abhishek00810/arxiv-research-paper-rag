import sys
sys.path.insert(0, 'src')
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import agent

#request schema
class QueryRequest(BaseModel):
    question: str

#response schema
class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retries: int

app = FastAPI()


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = agent.invoke({
            "query": request.question,
            "original_query": "",
            "retrieved_chunks": [],
            "relevance_decision": "",
            "relevance_reason": "",
            "retry_count": 0,
            "final_answer": "",
            "sources": []
        })
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"All LLM providers unavailable: {str(e)}")

    return QueryResponse(
        answer=result["final_answer"],
        sources=result["sources"],
        retries=result["retry_count"]
    )


@app.get("/health")
async def health():
    return {"status": "ok"}