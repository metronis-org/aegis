"""
Integration tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient

from metronis.api.main import app
from metronis.db.models import OrganizationModel
from metronis.db.session import SessionLocal


@pytest.fixture
def client():
    """Test client."""
    return TestClient(app)


@pytest.fixture
def api_key(db_session):
    """Create test organization and return API key."""
    org = OrganizationModel(
        name="Test Org",
        api_key="metronis_test123456789",
    )
    db_session.add(org)
    db_session.commit()
    return "metronis_test123456789"


@pytest.fixture
def db_session():
    """Database session for tests."""
    session = SessionLocal()
    yield session
    session.close()


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_check(self, client):
        """Test /health/ready endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestTraceEndpoints:
    """Test trace API endpoints."""

    def test_create_trace(self, client, api_key):
        """Test POST /api/v1/traces."""
        response = client.post(
            "/api/v1/traces",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4",
                "input": "What is 2+2?",
                "output": "4",
                "domain": "healthcare",
            },
        )
        assert response.status_code == 202
        assert "trace_id" in response.json()

    def test_create_trace_unauthorized(self, client):
        """Test creating trace without API key."""
        response = client.post(
            "/api/v1/traces",
            json={
                "model": "gpt-4",
                "input": "test",
                "output": "test",
                "domain": "healthcare",
            },
        )
        assert response.status_code == 401


class TestComplianceEndpoints:
    """Test compliance API endpoints."""

    def test_get_fda_report(self, client, api_key):
        """Test GET /api/v1/compliance/fda-tplc."""
        response = client.get(
            "/api/v1/compliance/fda-tplc",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200
        assert response.json()["report_type"] == "FDA_TPLC"
