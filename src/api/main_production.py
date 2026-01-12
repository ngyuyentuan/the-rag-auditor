"""
RAG Auditor Production API

Features:
- Error handling with structured responses
- Input validation
- API key authentication
- Rate limiting
- Structured logging
- Health checks
"""
import os
import time
import logging
import hashlib
import secrets
from datetime import datetime
from typing import Optional, List, Literal
from functools import wraps

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("rag_auditor_api")

# =============================================================================
# Configuration
# =============================================================================

class Config:
    API_KEYS = set(os.getenv("RAG_AUDITOR_API_KEYS", "demo-key-12345").split(","))
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    MAX_CLAIM_LENGTH = 500
    MAX_EVIDENCE_LENGTH = 2000
    MODEL_NAME = os.getenv("RAG_MODEL", "facebook/bart-large-mnli")

# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimiter:
    def __init__(self):
        self.requests = {}  # {api_key: [(timestamp, count)]}
    
    def is_allowed(self, api_key: str) -> bool:
        now = time.time()
        window_start = now - Config.RATE_LIMIT_WINDOW
        
        if api_key not in self.requests:
            self.requests[api_key] = []
        
        # Clean old entries
        self.requests[api_key] = [
            (ts, c) for ts, c in self.requests[api_key] 
            if ts > window_start
        ]
        
        # Count requests in window
        total = sum(c for _, c in self.requests[api_key])
        
        if total >= Config.RATE_LIMIT_REQUESTS:
            return False
        
        self.requests[api_key].append((now, 1))
        return True
    
    def get_remaining(self, api_key: str) -> int:
        if api_key not in self.requests:
            return Config.RATE_LIMIT_REQUESTS
        
        now = time.time()
        window_start = now - Config.RATE_LIMIT_WINDOW
        total = sum(c for ts, c in self.requests[api_key] if ts > window_start)
        return max(0, Config.RATE_LIMIT_REQUESTS - total)

rate_limiter = RateLimiter()

# =============================================================================
# Request/Response Models
# =============================================================================

class AuditRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=Config.MAX_CLAIM_LENGTH)
    evidence: str = Field(..., min_length=1, max_length=Config.MAX_EVIDENCE_LENGTH)
    
    @validator('claim')
    def validate_claim(cls, v):
        if not v.strip():
            raise ValueError('Claim cannot be empty')
        return v.strip()
    
    @validator('evidence')
    def validate_evidence(cls, v):
        if not v.strip():
            raise ValueError('Evidence cannot be empty')
        return v.strip()

class AuditResponse(BaseModel):
    success: bool
    verdict: str
    confidence: float
    explanation: str
    latency_ms: float
    request_id: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: str
    request_id: str

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    uptime_seconds: float

# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail={"error": "Missing API key", "error_code": "MISSING_API_KEY"}
        )
    if x_api_key not in Config.API_KEYS:
        logger.warning(f"Invalid API key attempt: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=401,
            detail={"error": "Invalid API key", "error_code": "INVALID_API_KEY"}
        )
    return x_api_key

# =============================================================================
# Application
# =============================================================================

app = FastAPI(
    title="RAG Auditor API",
    description="Production API for verifying AI claims against evidence",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Mount static files for Web UI
from fastapi.staticfiles import StaticFiles
static_path = Path(__file__).parent.parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Global state
START_TIME = time.time()
auditor = None

def get_auditor():
    global auditor
    if auditor is None:
        from src.auditor.rag_auditor_ultimate import RAGAuditorUltimate
        logger.info("Loading RAG Auditor model...")
        auditor = RAGAuditorUltimate(model=Config.MODEL_NAME)
        logger.info("Model loaded successfully")
    return auditor

def generate_request_id():
    return hashlib.md5(f"{time.time()}{secrets.token_hex(8)}".encode()).hexdigest()[:16]

# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = generate_request_id()
    logger.error(f"[{request_id}] Unhandled error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "request_id": request_id,
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = generate_request_id()
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail.get("error", str(exc.detail)) if isinstance(exc.detail, dict) else str(exc.detail),
            "error_code": exc.detail.get("error_code", "HTTP_ERROR") if isinstance(exc.detail, dict) else "HTTP_ERROR",
            "request_id": request_id,
        }
    )

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    return {"message": "RAG Auditor API", "version": "1.0.0", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=auditor is not None,
        uptime_seconds=time.time() - START_TIME,
    )

@app.post("/audit", response_model=AuditResponse)
async def audit(
    request: AuditRequest,
    api_key: str = Depends(verify_api_key)
):
    request_id = generate_request_id()
    
    # Rate limiting
    if not rate_limiter.is_allowed(api_key):
        remaining = rate_limiter.get_remaining(api_key)
        raise HTTPException(
            status_code=429,
            detail={
                "error": f"Rate limit exceeded. Try again in {Config.RATE_LIMIT_WINDOW}s",
                "error_code": "RATE_LIMITED",
                "remaining": remaining,
            }
        )
    
    logger.info(f"[{request_id}] Audit request: claim='{request.claim[:50]}...'")
    
    try:
        aud = get_auditor()
        result = aud.audit(request.claim, request.evidence)
        
        logger.info(f"[{request_id}] Result: {result.verdict} ({result.confidence:.2%})")
        
        return AuditResponse(
            success=True,
            verdict=result.verdict,
            confidence=result.confidence,
            explanation=result.explanation,
            latency_ms=result.latency_ms,
            request_id=request_id,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Audit failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Audit processing failed", "error_code": "AUDIT_FAILED"}
        )

@app.get("/stats")
async def stats(api_key: str = Depends(verify_api_key)):
    aud = get_auditor()
    return {
        "uptime_seconds": time.time() - START_TIME,
        "rate_limit_remaining": rate_limiter.get_remaining(api_key),
        "model": Config.MODEL_NAME,
    }

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG Auditor API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
