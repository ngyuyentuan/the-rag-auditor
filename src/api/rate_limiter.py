import time
from collections import defaultdict
from typing import Dict, Optional
from dataclasses import dataclass
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10


class RateLimiter:
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.minute_counts: Dict[str, list] = defaultdict(list)
        self.hour_counts: Dict[str, list] = defaultdict(list)
        self.day_counts: Dict[str, list] = defaultdict(list)
    
    def _cleanup(self, timestamps: list, window: float) -> list:
        now = time.time()
        return [t for t in timestamps if now - t < window]
    
    def check_rate_limit(self, client_id: str) -> bool:
        now = time.time()
        
        self.minute_counts[client_id] = self._cleanup(self.minute_counts[client_id], 60)
        if len(self.minute_counts[client_id]) >= self.config.requests_per_minute:
            return False
        
        self.hour_counts[client_id] = self._cleanup(self.hour_counts[client_id], 3600)
        if len(self.hour_counts[client_id]) >= self.config.requests_per_hour:
            return False
        
        self.day_counts[client_id] = self._cleanup(self.day_counts[client_id], 86400)
        if len(self.day_counts[client_id]) >= self.config.requests_per_day:
            return False
        
        self.minute_counts[client_id].append(now)
        self.hour_counts[client_id].append(now)
        self.day_counts[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> Dict[str, int]:
        self.minute_counts[client_id] = self._cleanup(self.minute_counts[client_id], 60)
        self.hour_counts[client_id] = self._cleanup(self.hour_counts[client_id], 3600)
        self.day_counts[client_id] = self._cleanup(self.day_counts[client_id], 86400)
        return {
            "minute": max(0, self.config.requests_per_minute - len(self.minute_counts[client_id])),
            "hour": max(0, self.config.requests_per_hour - len(self.hour_counts[client_id])),
            "day": max(0, self.config.requests_per_day - len(self.day_counts[client_id])),
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limiter: RateLimiter = None):
        super().__init__(app)
        self.limiter = rate_limiter or RateLimiter()
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        api_key = request.headers.get("X-API-Key", "")
        client_id = api_key if api_key else client_ip
        
        if not self.limiter.check_rate_limit(client_id):
            remaining = self.limiter.get_remaining(client_id)
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "remaining": remaining,
                    "retry_after": 60
                }
            )
        
        response = await call_next(request)
        remaining = self.limiter.get_remaining(client_id)
        response.headers["X-RateLimit-Remaining-Minute"] = str(remaining["minute"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(remaining["hour"])
        return response
