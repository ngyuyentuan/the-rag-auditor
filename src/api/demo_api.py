"""
RAG Auditor - Full Demo API with Knowledge Upload

Features:
- Upload knowledge files (txt, pdf, docx)
- In-memory knowledge storage
- Simple text search retrieval
- Audit AI responses against retrieved evidence
"""
import os
import time
import logging
import hashlib
import secrets
import re
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Header, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("rag_auditor")

# =============================================================================
# Configuration
# =============================================================================

class Config:
    API_KEYS = set(os.getenv("RAG_AUDITOR_API_KEYS", "demo-key-12345").split(","))
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.doc'}


# =============================================================================
# File Parsers
# =============================================================================

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        from PyPDF2 import PdfReader
        import io
        reader = PdfReader(io.BytesIO(content))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to parse PDF: {str(e)}")


def extract_text_from_docx(content: bytes) -> str:
    """Extract text from DOCX bytes."""
    try:
        from docx import Document
        import io
        doc = Document(io.BytesIO(content))
        text_parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to parse DOCX: {str(e)}")

# =============================================================================
# Knowledge Store (In-Memory with Semantic Search)
# =============================================================================

# Global embedder for semantic search
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading multilingual embedder...")
        _embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("Embedder loaded")
    return _embedder


class KnowledgeStore:
    def __init__(self):
        self.documents: Dict[str, Dict] = {}
        self.chunks: List[Dict] = []
        self.embeddings = None  # numpy array of embeddings
    
    def add_document(self, name: str, content: str) -> str:
        doc_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]
        
        # Split into chunks
        paragraphs = [p.strip() for p in content.split('\n') if p.strip() and len(p.strip()) > 20]
        if not paragraphs:
            paragraphs = [content[:1000]]
        
        # Limit chunk size
        chunks = []
        for i, para in enumerate(paragraphs[:100]):  # Max 100 chunks per doc
            chunk_id = f"{doc_id}_{i}"
            chunks.append({"doc_id": doc_id, "chunk_id": chunk_id, "text": para[:500]})
        
        self.documents[doc_id] = {"name": name, "content": content[:5000], "chunks": len(chunks)}
        self.chunks.extend(chunks)
        
        # Compute embeddings for new chunks
        embedder = get_embedder()
        new_texts = [c["text"] for c in chunks]
        new_embs = embedder.encode(new_texts)
        
        import numpy as np
        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])
        
        logger.info(f"Added document: {name} ({len(chunks)} chunks)")
        return doc_id
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Semantic search using multilingual embeddings."""
        if not self.chunks or self.embeddings is None:
            return []
        
        import numpy as np
        embedder = get_embedder()
        query_emb = embedder.encode([query])[0]
        
        # Compute similarities
        similarities = np.dot(self.embeddings, query_emb) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Similarity threshold
                results.append(self.chunks[idx]["text"])
        
        return results
    
    def get_stats(self) -> Dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "doc_names": [d["name"] for d in self.documents.values()]
        }
    
    def clear(self):
        self.documents.clear()
        self.chunks.clear()
        self.embeddings = None


knowledge_store = KnowledgeStore()

# =============================================================================
# Models
# =============================================================================

class AuditRequest(BaseModel):
    ai_response: str = Field(..., min_length=1, max_length=2000)
    query: Optional[str] = Field(None, max_length=500)
    evidence: Optional[str] = Field(None, max_length=5000)

class AuditResponse(BaseModel):
    success: bool
    verdict: str
    confidence: float
    explanation: str
    retrieved_evidence: str
    latency_ms: float

# =============================================================================
# Authentication
# =============================================================================

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None or x_api_key not in Config.API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# =============================================================================
# Application
# =============================================================================

app = FastAPI(title="RAG Auditor Demo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_path = ROOT / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

START_TIME = time.time()
auditor = None

def get_auditor():
    global auditor
    if auditor is None:
        from src.auditor.multilingual_v2 import MultilingualAuditorV2
        logger.info("Loading Multilingual RAG Auditor V2 (Vietnamese + English)...")
        auditor = MultilingualAuditorV2(use_persistent_cache=True)
        logger.info(f"Auditor v{auditor.VERSION} loaded")
    return auditor

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    return {"message": "RAG Auditor Demo API", "docs": "/docs", "ui": "/static/index.html"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime": time.time() - START_TIME,
        "knowledge": knowledge_store.get_stats()
    }

@app.post("/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Upload a knowledge file (txt, md, pdf, docx)."""
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"File type {ext} not allowed. Use: {list(Config.ALLOWED_EXTENSIONS)}")
    
    # Read content
    content = await file.read()
    if len(content) > Config.MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Max: {Config.MAX_FILE_SIZE // 1024 // 1024}MB")
    
    # Extract text based on file type
    try:
        if ext == '.pdf':
            text = extract_text_from_pdf(content)
        elif ext in ('.docx', '.doc'):
            text = extract_text_from_docx(content)
        else:
            # Plain text files
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse file: {str(e)}")
    
    if not text.strip():
        raise HTTPException(400, "No text content found in file")
    
    # Store
    doc_id = knowledge_store.add_document(file.filename, text)
    
    return {
        "success": True,
        "doc_id": doc_id,
        "filename": file.filename,
        "file_type": ext,
        "chunks": knowledge_store.documents[doc_id]["chunks"],
        "message": f"File uploaded successfully with {knowledge_store.documents[doc_id]['chunks']} chunks"
    }

@app.get("/knowledge")
async def get_knowledge(api_key: str = Depends(verify_api_key)):
    """Get uploaded knowledge info."""
    return knowledge_store.get_stats()

@app.delete("/knowledge")
async def clear_knowledge(api_key: str = Depends(verify_api_key)):
    """Clear all uploaded knowledge."""
    knowledge_store.clear()
    return {"success": True, "message": "Knowledge cleared"}

@app.post("/audit", response_model=AuditResponse)
async def audit(
    request: AuditRequest,
    api_key: str = Depends(verify_api_key)
):
    """Audit an AI response against knowledge base."""
    start = time.time()
    
    # Get evidence: either provided or retrieved from knowledge
    if request.evidence:
        evidence = request.evidence
    elif knowledge_store.chunks:
        query = request.query or request.ai_response
        retrieved = knowledge_store.search(query, top_k=3)
        evidence = "\n\n".join(retrieved) if retrieved else "No relevant evidence found."
    else:
        raise HTTPException(400, "No evidence provided and no knowledge uploaded. Please upload knowledge files first.")
    
    # Audit
    aud = get_auditor()
    result = aud.audit(request.ai_response, evidence)
    
    return AuditResponse(
        success=True,
        verdict=result.verdict,
        confidence=result.confidence,
        explanation=result.explanation,
        retrieved_evidence=evidence[:500] + "..." if len(evidence) > 500 else evidence,
        latency_ms=(time.time() - start) * 1000
    )


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    groq_key: Optional[str] = None
    openai_key: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    verified: bool
    verdict: str
    confidence: float
    provider: str = "Template"


class SettingsRequest(BaseModel):
    groq_key: Optional[str] = None
    openai_key: Optional[str] = None


# Global LLM settings
_llm_settings = {"groq_key": "", "openai_key": ""}


@app.post("/settings")
async def update_settings(request: SettingsRequest, api_key: str = Depends(verify_api_key)):
    """Update LLM API keys."""
    if request.groq_key:
        _llm_settings["groq_key"] = request.groq_key
    if request.openai_key:
        _llm_settings["openai_key"] = request.openai_key
    return {"success": True, "message": "Settings updated"}


@app.get("/providers")
async def get_providers(api_key: str = Depends(verify_api_key)):
    """Get available LLM providers status."""
    return {
        "providers": [
            {"name": "Groq (Free)", "available": bool(_llm_settings.get("groq_key")), "type": "cloud"},
            {"name": "OpenAI", "available": bool(_llm_settings.get("openai_key")), "type": "cloud"},
            {"name": "Ollama", "available": True, "type": "local"},
            {"name": "Template", "available": True, "type": "fallback"},
        ],
        "active_keys": {
            "groq": bool(_llm_settings.get("groq_key")),
            "openai": bool(_llm_settings.get("openai_key"))
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_with_knowledge(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    """Chat endpoint with multi-LLM provider support."""
    if not knowledge_store.chunks:
        raise HTTPException(400, "No knowledge uploaded. Please upload files first.")
    
    # Retrieve relevant knowledge
    retrieved = knowledge_store.search(request.question, top_k=5)
    if not retrieved:
        return ChatResponse(
            answer="I couldn't find relevant information in the knowledge base.",
            sources=[],
            verified=False,
            verdict="NEI",
            confidence=0.0,
            provider="None"
        )
    
    context = "\n\n".join(retrieved)
    
    # Import LLM provider
    from src.llm.providers import generate_answer
    
    # Get API keys from request or global settings
    groq_key = request.groq_key or _llm_settings.get("groq_key", "")
    openai_key = request.openai_key or _llm_settings.get("openai_key", "")
    
    # Generate answer using best available provider
    answer, provider = await generate_answer(
        question=request.question,
        context=context,
        groq_key=groq_key,
        openai_key=openai_key
    )
    
    # Verify the answer against evidence
    aud = get_auditor()
    result = aud.audit(answer, context)
    
    return ChatResponse(
        answer=answer,
        sources=retrieved[:2],
        verified=True,
        verdict=result.verdict,
        confidence=result.confidence,
        provider=provider
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
