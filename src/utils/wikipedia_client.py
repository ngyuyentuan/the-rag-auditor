import requests
import time
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

logger = logging.getLogger("wikipedia")

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "wiki_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class WikiArticle:
    title: str
    text: str
    summary: str
    url: str


class WikipediaClient:
    BASE_URL = "https://en.wikipedia.org/api/rest_v1"
    API_URL = "https://en.wikipedia.org/w/api.php"
    
    def __init__(self, cache_enabled: bool = True, rate_limit_ms: int = 100):
        self.cache_enabled = cache_enabled
        self.rate_limit_ms = rate_limit_ms
        self._last_request = 0
        self._cache = {}
    
    def _rate_limit(self):
        elapsed = (time.time() - self._last_request) * 1000
        if elapsed < self.rate_limit_ms:
            time.sleep((self.rate_limit_ms - elapsed) / 1000)
        self._last_request = time.time()
    
    def _cache_key(self, title: str) -> str:
        return hashlib.md5(title.lower().encode()).hexdigest()
    
    def _get_cache_path(self, title: str) -> Path:
        return CACHE_DIR / f"{self._cache_key(title)}.json"
    
    def _load_from_cache(self, title: str) -> Optional[WikiArticle]:
        if not self.cache_enabled:
            return None
        
        key = self._cache_key(title)
        if key in self._cache:
            return self._cache[key]
        
        cache_path = self._get_cache_path(title)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                article = WikiArticle(**data)
                self._cache[key] = article
                return article
            except:
                pass
        return None
    
    def _save_to_cache(self, title: str, article: WikiArticle):
        if not self.cache_enabled:
            return
        
        key = self._cache_key(title)
        self._cache[key] = article
        
        cache_path = self._get_cache_path(title)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "title": article.title,
                    "text": article.text,
                    "summary": article.summary,
                    "url": article.url
                }, f)
        except:
            pass
    
    def get_article(self, title: str) -> Optional[WikiArticle]:
        cached = self._load_from_cache(title)
        if cached:
            return cached
        
        self._rate_limit()
        
        clean_title = title.replace("_", " ").strip()
        
        try:
            params = {
                "action": "query",
                "titles": clean_title,
                "prop": "extracts|info",
                "explaintext": True,
                "inprop": "url",
                "format": "json",
                "redirects": 1
            }
            
            headers = {
                "User-Agent": "RAGAuditor/1.0 (Academic Research; contact@example.com)"
            }
            
            response = requests.get(self.API_URL, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get("query", {}).get("pages", {})
            for page_id, page in pages.items():
                if page_id == "-1":
                    return None
                
                article = WikiArticle(
                    title=page.get("title", clean_title),
                    text=page.get("extract", ""),
                    summary=page.get("extract", "")[:500] if page.get("extract") else "",
                    url=page.get("fullurl", f"https://en.wikipedia.org/wiki/{title}")
                )
                
                if article.text:
                    self._save_to_cache(title, article)
                    return article
            
        except Exception as e:
            logger.warning(f"Failed to fetch {title}: {e}")
        
        return None
    
    def get_articles_batch(self, titles: List[str], max_workers: int = 5) -> Dict[str, WikiArticle]:
        results = {}
        for title in titles:
            article = self.get_article(title)
            if article:
                results[title] = article
        return results


if __name__ == "__main__":
    client = WikipediaClient()
    
    test_titles = [
        "Telemundo",
        "Damon_Albarn",
        "Albert_Einstein",
        "Fox_2000_Pictures",
    ]
    
    for title in test_titles:
        print(f"\n{'='*50}")
        print(f"Fetching: {title}")
        article = client.get_article(title)
        if article:
            print(f"Title: {article.title}")
            print(f"Summary: {article.summary[:200]}...")
        else:
            print("Not found")
