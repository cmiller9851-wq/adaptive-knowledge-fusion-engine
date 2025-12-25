from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from .engine import FusionEngine

app = FastAPI(title="Adaptive Knowledge Fusion Engine API", version="0.2.0")

engine = FusionEngine()   # loads config, indexes, etc.

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    stream: bool = False

class DocumentResponse(BaseModel):
    doc_id: str
    title: str
    snippet: str
    source: str
    score: float
    provenance: Dict[str, Any]

@app.post("/query", response_model=List[DocumentResponse])
async def query(req: QueryRequest):
    try:
        results = engine.query(req.query, top_k=req.top_k)
        return [
            DocumentResponse(
                doc_id=r.doc_id,
                title=r.title,
                snippet=r.snippet,
                source=r.source,
                score=r.score,
                provenance=r.provenance,
            )
            for r in results
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
