# RAG Auditor

AI Claim Verification System - Verify AI-generated claims against evidence.

## Quick Start

### Python API
```python
from src.auditor.rag_auditor_ultimate import RAGAuditorUltimate

auditor = RAGAuditorUltimate()
result = auditor.audit("Claim text", "Evidence text")
print(result.verdict)  # SUPPORTS / REFUTES / NEI
```

### REST API
```bash
# Start server
python -m uvicorn src.api.main_production:app --reload

# Make request
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -H "X-API-Key: demo-key-12345" \
  -d '{"claim":"The earth is flat","evidence":"Earth is a sphere"}'
```

### Docker
```bash
docker-compose up -d
```

### Web UI
Open http://localhost:8000/static/index.html

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 87% |
| SUPPORTS | 100% |
| REFUTES | 100% |
| NEI | 63% |
| Speed | 135ms |

## Security

- API Key authentication
- Rate limiting (100 req/min)
- Input validation
- Structured logging

## License

MIT
