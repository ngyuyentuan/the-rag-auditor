import os
import time
import logging
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger("auth")


@dataclass
class TokenPayload:
    user_id: str
    exp: float
    iat: float
    scope: str = "api"


class SimpleJWT:
    def __init__(self, secret_key: str = None, token_expiry_hours: int = 24):
        self.secret_key = secret_key or os.getenv("JWT_SECRET", "change-this-secret-in-production")
        self.token_expiry_hours = token_expiry_hours
    
    def _sign(self, data: str) -> str:
        return hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def create_token(self, user_id: str, scope: str = "api") -> str:
        now = time.time()
        exp = now + (self.token_expiry_hours * 3600)
        payload = f"{user_id}:{exp}:{now}:{scope}"
        signature = self._sign(payload)
        return f"{payload}:{signature}"
    
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        try:
            parts = token.split(":")
            if len(parts) != 5:
                return None
            user_id, exp, iat, scope, signature = parts
            payload = f"{user_id}:{exp}:{iat}:{scope}"
            expected_sig = self._sign(payload)
            if not hmac.compare_digest(signature, expected_sig):
                logger.warning(f"Invalid signature for token")
                return None
            if float(exp) < time.time():
                logger.info(f"Token expired for user {user_id}")
                return None
            return TokenPayload(
                user_id=user_id,
                exp=float(exp),
                iat=float(iat),
                scope=scope
            )
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None


class APIKeyAuth:
    def __init__(self, valid_keys: set = None):
        default_keys = os.getenv("API_KEYS", "demo-key-12345").split(",")
        self.valid_keys = valid_keys or set(default_keys)
    
    def verify_key(self, api_key: str) -> bool:
        return api_key in self.valid_keys
    
    def add_key(self, api_key: str):
        self.valid_keys.add(api_key)
    
    def remove_key(self, api_key: str):
        self.valid_keys.discard(api_key)


if __name__ == "__main__":
    jwt = SimpleJWT(secret_key="test-secret")
    token = jwt.create_token("user123", "admin")
    print(f"Token: {token}")
    payload = jwt.verify_token(token)
    print(f"Payload: {payload}")
    
    auth = APIKeyAuth()
    print(f"Valid key: {auth.verify_key('demo-key-12345')}")
    print(f"Invalid key: {auth.verify_key('wrong-key')}")
