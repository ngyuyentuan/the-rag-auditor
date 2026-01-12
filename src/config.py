import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class AuditorConfig:
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    device: str = "auto"
    enable_cache: bool = True
    nei_same_text_check: bool = False
    
    @classmethod
    def from_env(cls) -> "AuditorConfig":
        return cls(
            nli_model=os.getenv("NLI_MODEL", cls.nli_model),
            device=os.getenv("DEVICE", cls.device),
            enable_cache=os.getenv("ENABLE_CACHE", "true").lower() == "true",
            nei_same_text_check=os.getenv("NEI_SAME_TEXT_CHECK", "false").lower() == "true",
        )


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    rate_limit_per_day: int = 10000
    
    jwt_secret: str = "change-this-in-production"
    jwt_expiry_hours: int = 24
    
    api_keys: List[str] = field(default_factory=lambda: ["demo-key-12345"])
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        api_keys_str = os.getenv("API_KEYS", "demo-key-12345")
        return cls(
            host=os.getenv("API_HOST", cls.host),
            port=int(os.getenv("API_PORT", cls.port)),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_MINUTE", cls.rate_limit_per_minute)),
            rate_limit_per_hour=int(os.getenv("RATE_LIMIT_HOUR", cls.rate_limit_per_hour)),
            rate_limit_per_day=int(os.getenv("RATE_LIMIT_DAY", cls.rate_limit_per_day)),
            jwt_secret=os.getenv("JWT_SECRET", cls.jwt_secret),
            jwt_expiry_hours=int(os.getenv("JWT_EXPIRY_HOURS", cls.jwt_expiry_hours)),
            api_keys=api_keys_str.split(","),
        )


@dataclass
class DatabaseConfig:
    driver: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "rag_auditor"
    user: str = "postgres"
    password: str = ""
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            driver=os.getenv("DB_DRIVER", cls.driver),
            host=os.getenv("DB_HOST", cls.host),
            port=int(os.getenv("DB_PORT", cls.port)),
            database=os.getenv("DB_DATABASE", cls.database),
            user=os.getenv("DB_USER", cls.user),
            password=os.getenv("DB_PASSWORD", cls.password),
        )
    
    @property
    def connection_string(self) -> str:
        if self.driver == "sqlite":
            return f"sqlite:///{self.database}.db"
        return f"{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class Config:
    auditor: AuditorConfig = field(default_factory=AuditorConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    log_level: str = "INFO"
    log_dir: str = "logs"
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            auditor=AuditorConfig.from_env(),
            api=APIConfig.from_env(),
            database=DatabaseConfig.from_env(),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_dir=os.getenv("LOG_DIR", "logs"),
        )


config = Config.from_env()


if __name__ == "__main__":
    print(f"Auditor: {config.auditor}")
    print(f"API: {config.api}")
    print(f"Database: {config.database.connection_string}")
