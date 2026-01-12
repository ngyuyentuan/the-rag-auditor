import os
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class StructuredLogger:
    def __init__(self, name: str, level: str = None):
        self.logger = logging.getLogger(name)
        level = level or os.getenv("LOG_LEVEL", "INFO")
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(handler)
    
    def _log(self, level: int, message: str, **kwargs):
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0, message, (), None
        )
        record.extra_data = kwargs
        self.logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)


def setup_file_logging(log_dir: Path, max_bytes: int = 10_000_000, backup_count: int = 5):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"
    
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    handler.setFormatter(StructuredFormatter())
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    return log_file


if __name__ == "__main__":
    logger = get_logger("test")
    logger.info("Application started", version="1.0.0")
    logger.info("Request received", endpoint="/audit", method="POST")
    logger.warning("High latency detected", latency_ms=500)
    logger.error("Failed to process", error_code="E001", claim="test")
