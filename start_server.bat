@echo off
REM RAG Auditor - Start Production Server

echo Starting RAG Auditor API...
echo.
echo API Docs: http://localhost:8000/docs
echo Web UI: http://localhost:8000/static/index.html
echo.
echo Default API Key: demo-key-12345
echo.

python -m uvicorn src.api.main_production:app --host 0.0.0.0 --port 8000
