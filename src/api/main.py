"""
RAG Auditor API - FastAPI Production Endpoints

Production-ready API for claim verification.

Run with: uvicorn src.api.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Literal
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auditor.rag_auditor_pro import RAGAuditorPro, AuditResult

app = FastAPI(
    title="RAG Auditor API",
    description="Production API for verifying AI-generated claims against evidence",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global auditor instances for each mode
auditors = {}


def get_auditor(mode: str = "balanced") -> RAGAuditorPro:
    if mode not in auditors:
        auditors[mode] = RAGAuditorPro(mode=mode)
    return auditors[mode]


class AuditRequest(BaseModel):
    claim: str
    evidence: str
    mode: Literal["safety", "balanced", "coverage"] = "balanced"


class AuditResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    explanation: str
    latency_ms: float
    mode: str


class BatchAuditRequest(BaseModel):
    pairs: List[dict]  # [{"claim": "...", "evidence": "..."}]
    mode: Literal["safety", "balanced", "coverage"] = "balanced"


class HealthResponse(BaseModel):
    status: str
    version: str
    modes_loaded: List[str]


@app.get("/")
async def root():
    return {"message": "RAG Auditor API", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        modes_loaded=list(auditors.keys()),
    )


@app.post("/audit", response_model=AuditResponse)
async def audit(request: AuditRequest):
    """
    Audit a single claim against evidence.
    
    - **claim**: The statement to verify
    - **evidence**: The source text to check against
    - **mode**: safety (catch hallucinations), balanced (default), coverage (minimize NEI)
    """
    try:
        auditor = get_auditor(request.mode)
        result = auditor.audit(request.claim, request.evidence)
        
        return AuditResponse(
            claim=result.claim,
            verdict=result.verdict,
            confidence=result.confidence,
            explanation=result.explanation,
            latency_ms=result.latency_ms,
            mode=result.mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audit/batch")
async def audit_batch(request: BatchAuditRequest):
    """Batch audit multiple claim-evidence pairs."""
    try:
        auditor = get_auditor(request.mode)
        pairs = [(p["claim"], p["evidence"]) for p in request.pairs]
        results = auditor.audit_batch(pairs)
        
        return {
            "results": [
                {
                    "claim": r.claim,
                    "verdict": r.verdict,
                    "confidence": r.confidence,
                    "explanation": r.explanation,
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """Get usage statistics for all modes."""
    return {mode: a.get_stats() for mode, a in auditors.items()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
