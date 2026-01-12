"""
Integration Tests for RAG Auditor API
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.demo_api import app


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def api_key(self):
        return "demo-key-12345"
    
    # =====================================================================
    # Health & Info Endpoints
    # =====================================================================
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data
    
    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime" in data
    
    # =====================================================================
    # Upload Endpoints
    # =====================================================================
    
    def test_upload_text_file(self, client, api_key):
        """Test uploading a text file."""
        content = b"This is test content for the knowledge base."
        response = client.post(
            "/upload",
            files={"file": ("test.txt", content, "text/plain")},
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "chunks" in data
    
    def test_upload_without_api_key(self, client):
        """Test upload fails without API key."""
        content = b"Test content"
        response = client.post(
            "/upload",
            files={"file": ("test.txt", content, "text/plain")}
        )
        assert response.status_code in [401, 403]
    
    # =====================================================================
    # Chat Endpoint
    # =====================================================================
    
    def test_chat_basic(self, client, api_key):
        """Test basic chat functionality."""
        # First upload some content
        content = b"Vaccines are safe and effective according to research."
        client.post(
            "/upload",
            files={"file": ("vaccines.txt", content, "text/plain")},
            headers={"X-API-Key": api_key}
        )
        
        # Then ask a question
        response = client.post(
            "/chat",
            json={"question": "Are vaccines safe?"},
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "verdict" in data
        assert "confidence" in data
    
    def test_chat_without_knowledge(self, client, api_key):
        """Test chat without knowledge base."""
        # Clear knowledge first
        client.delete("/knowledge", headers={"X-API-Key": api_key})
        
        response = client.post(
            "/chat",
            json={"question": "Random question?"},
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
    
    # =====================================================================
    # Knowledge Management
    # =====================================================================
    
    def test_knowledge_stats(self, client, api_key):
        """Test knowledge stats endpoint."""
        response = client.get(
            "/knowledge",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "chunks" in data
    
    def test_delete_knowledge(self, client, api_key):
        """Test clearing knowledge base."""
        response = client.delete(
            "/knowledge",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
    
    # =====================================================================
    # Settings
    # =====================================================================
    
    def test_settings_endpoint(self, client, api_key):
        """Test settings endpoint."""
        response = client.post(
            "/settings",
            json={"groq_key": "test", "openai_key": ""},
            headers={"X-API-Key": api_key, "Content-Type": "application/json"}
        )
        assert response.status_code == 200
    
    def test_providers_endpoint(self, client, api_key):
        """Test providers endpoint."""
        response = client.get(
            "/providers",
            headers={"X-API-Key": api_key}
        )
        assert response.status_code == 200
        data = response.json()
        assert "providers" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
