import time
import threading
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger("metrics")


@dataclass
class MetricValue:
    total: float = 0.0
    count: int = 0
    min_val: float = float('inf')
    max_val: float = float('-inf')
    
    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
    
    def record(self, value: float):
        self.total += value
        self.count += 1
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)


class PrometheusMetrics:
    def __init__(self):
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, MetricValue] = defaultdict(MetricValue)
        self._labels: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._start_time = time.time()
    
    def inc(self, name: str, value: int = 1, labels: Dict[str, str] = None):
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] += value
            if labels:
                self._labels[key] = labels
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
            if labels:
                self._labels[key] = labels
    
    def observe(self, name: str, value: float, labels: Dict[str, str] = None):
        key = self._make_key(name, labels)
        with self._lock:
            self._histograms[key].record(value)
            if labels:
                self._labels[key] = labels
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_prometheus_format(self) -> str:
        lines = []
        lines.append(f"# Generated at {datetime.now().isoformat()}")
        lines.append(f"# Uptime: {time.time() - self._start_time:.2f}s")
        lines.append("")
        
        with self._lock:
            for key, value in self._counters.items():
                lines.append(f"# TYPE {key.split('{')[0]} counter")
                lines.append(f"{key} {value}")
            
            for key, value in self._gauges.items():
                lines.append(f"# TYPE {key.split('{')[0]} gauge")
                lines.append(f"{key} {value}")
            
            for key, metric in self._histograms.items():
                base_name = key.split('{')[0]
                lines.append(f"# TYPE {base_name} summary")
                lines.append(f"{key}_count {metric.count}")
                lines.append(f"{key}_sum {metric.total}")
                lines.append(f"{key}_avg {metric.avg:.4f}")
                if metric.count > 0:
                    lines.append(f"{key}_min {metric.min_val:.4f}")
                    lines.append(f"{key}_max {metric.max_val:.4f}")
        
        return "\n".join(lines)
    
    def get_json_format(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "uptime_seconds": time.time() - self._start_time,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: {
                        "count": v.count,
                        "sum": v.total,
                        "avg": v.avg,
                        "min": v.min_val if v.count > 0 else None,
                        "max": v.max_val if v.count > 0 else None,
                    }
                    for k, v in self._histograms.items()
                }
            }


metrics = PrometheusMetrics()


class StructuredLogger:
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
    
    def _format(self, level: str, message: str, **kwargs) -> str:
        data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs
        }
        return json.dumps(data)
    
    def info(self, message: str, **kwargs):
        self.logger.info(self._format("INFO", message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format("WARNING", message, **kwargs))
    
    def error(self, message: str, **kwargs):
        self.logger.error(self._format("ERROR", message, **kwargs))
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format("DEBUG", message, **kwargs))


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)


if __name__ == "__main__":
    metrics.inc("requests_total")
    metrics.inc("requests_total", labels={"endpoint": "/audit"})
    metrics.set_gauge("active_connections", 5)
    metrics.observe("request_duration_seconds", 0.123)
    metrics.observe("request_duration_seconds", 0.456)
    
    print("Prometheus format:")
    print(metrics.get_prometheus_format())
    
    print("\n\nJSON format:")
    print(json.dumps(metrics.get_json_format(), indent=2))
