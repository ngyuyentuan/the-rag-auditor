import json
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading


@dataclass
class CacheEntry:
    key: str
    value: str
    created_at: float
    expires_at: Optional[float] = None


class PersistentCache:
    def __init__(self, db_path: Path = None, default_ttl: int = 86400 * 7):
        self.db_path = db_path or Path("data/cache.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
    
    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False)
    
    def _make_key(self, namespace: str, *args) -> str:
        data = f"{namespace}:" + ":".join(str(a) for a in args)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                if row:
                    value, expires_at = row
                    if expires_at is None or expires_at > time.time():
                        return value
                    else:
                        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        return None
    
    def set(self, key: str, value: str, ttl: int = None):
        ttl = ttl or self.default_ttl
        now = time.time()
        expires_at = now + ttl if ttl > 0 else None
        
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key, value, now, expires_at)
                )
    
    def delete(self, key: str):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
    
    def clear(self):
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM cache")
    
    def cleanup_expired(self) -> int:
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (time.time(),)
                )
                return cursor.rowcount
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            with self._get_conn() as conn:
                total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                expired = conn.execute(
                    "SELECT COUNT(*) FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?",
                    (time.time(),)
                ).fetchone()[0]
                return {
                    "total_entries": total,
                    "expired_entries": expired,
                    "db_path": str(self.db_path),
                }


class NLICache(PersistentCache):
    def __init__(self, db_path: Path = None):
        super().__init__(db_path or Path("data/nli_cache.db"), default_ttl=86400 * 30)
    
    def get_nli_result(self, claim: str, evidence: str) -> Optional[Dict[str, float]]:
        key = self._make_key("nli", claim, evidence)
        value = self.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set_nli_result(self, claim: str, evidence: str, probs: Dict[str, float]):
        key = self._make_key("nli", claim, evidence)
        self.set(key, json.dumps(probs))


class WikiCache(PersistentCache):
    def __init__(self, db_path: Path = None):
        super().__init__(db_path or Path("data/wiki_cache.db"), default_ttl=86400 * 7)
    
    def get_article(self, title: str) -> Optional[str]:
        key = self._make_key("wiki", title.lower())
        return self.get(key)
    
    def set_article(self, title: str, text: str):
        key = self._make_key("wiki", title.lower())
        self.set(key, text)


if __name__ == "__main__":
    cache = NLICache()
    
    cache.set_nli_result("test claim", "test evidence", {"entailment": 0.8, "contradiction": 0.1, "neutral": 0.1})
    
    result = cache.get_nli_result("test claim", "test evidence")
    print(f"Cached result: {result}")
    
    print(f"Stats: {cache.get_stats()}")
