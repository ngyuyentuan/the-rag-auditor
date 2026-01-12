"""
Persistent Knowledge Store using SQLite

Features:
1. Persistent document storage (survives restarts)
2. Chunk-level embedding caching
3. Efficient semantic search with FAISS-like indexing
4. Thread-safe operations
"""
import os
import json
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import threading

# Thread-local storage for connections
_local = threading.local()


class PersistentKnowledgeStore:
    """SQLite-based persistent knowledge store with semantic search."""
    
    def __init__(self, db_path: str = "data/knowledge.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = None
        self._embeddings_cache = None  # In-memory for fast search
        self._chunk_ids = []
        self._init_db()
        self._load_embeddings_cache()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(_local, 'connection') or _local.connection is None:
            _local.connection = sqlite3.connect(str(self.db_path), check_same_thread=False)
            _local.connection.row_factory = sqlite3.Row
        return _local.connection
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                content TEXT,
                chunks_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
            );
            
            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        """)
        conn.commit()
    
    def _get_embedder(self):
        """Lazy load embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        return self._embedder
    
    def _load_embeddings_cache(self):
        """Load all embeddings into memory for fast search."""
        conn = self._get_conn()
        cursor = conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()
        
        if rows:
            self._chunk_ids = [row['id'] for row in rows]
            embeddings = [np.frombuffer(row['embedding'], dtype=np.float32) for row in rows]
            self._embeddings_cache = np.stack(embeddings) if embeddings else None
        else:
            self._chunk_ids = []
            self._embeddings_cache = None
    
    def add_document(self, name: str, content: str) -> str:
        """Add document and compute embeddings."""
        doc_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]
        
        # Split into chunks
        paragraphs = [p.strip() for p in content.split('\n') if p.strip() and len(p.strip()) > 20]
        if not paragraphs:
            paragraphs = [content[:1000]] if content else ["Empty document"]
        
        conn = self._get_conn()
        
        # Insert document
        conn.execute(
            "INSERT INTO documents (id, name, content, chunks_count) VALUES (?, ?, ?, ?)",
            (doc_id, name, content[:10000], len(paragraphs))
        )
        
        # Insert chunks with embeddings
        embedder = self._get_embedder()
        chunk_texts = [p[:500] for p in paragraphs[:100]]
        embeddings = embedder.encode(chunk_texts)
        
        for i, (text, emb) in enumerate(zip(chunk_texts, embeddings)):
            chunk_id = f"{doc_id}_{i}"
            conn.execute(
                "INSERT INTO chunks (id, doc_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, doc_id, i, text, emb.astype(np.float32).tobytes())
            )
        
        conn.commit()
        
        # Update cache
        self._load_embeddings_cache()
        
        return doc_id
    
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """Semantic search using cached embeddings."""
        if self._embeddings_cache is None or len(self._embeddings_cache) == 0:
            return []
        
        embedder = self._get_embedder()
        query_emb = embedder.encode([query])[0]
        
        # Cosine similarity
        similarities = np.dot(self._embeddings_cache, query_emb) / (
            np.linalg.norm(self._embeddings_cache, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Fetch texts
        conn = self._get_conn()
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:
                chunk_id = self._chunk_ids[idx]
                cursor = conn.execute("SELECT text FROM chunks WHERE id = ?", (chunk_id,))
                row = cursor.fetchone()
                if row:
                    results.append(row['text'])
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        conn = self._get_conn()
        
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        docs = conn.execute("SELECT name FROM documents ORDER BY created_at DESC").fetchall()
        
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "doc_names": [d['name'] for d in docs]
        }
    
    def clear(self):
        """Clear all data."""
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM documents")
        conn.commit()
        self._embeddings_cache = None
        self._chunk_ids = []
    
    def delete_document(self, doc_id: str):
        """Delete a specific document."""
        conn = self._get_conn()
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        self._load_embeddings_cache()


# Singleton instance
_store = None

def get_knowledge_store(db_path: str = "data/knowledge.db") -> PersistentKnowledgeStore:
    """Get singleton knowledge store instance."""
    global _store
    if _store is None:
        _store = PersistentKnowledgeStore(db_path)
    return _store
