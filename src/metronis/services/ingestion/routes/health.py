"""Health check endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from metronis.infrastructure.database import get_database

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str = "0.1.0"
    database: str = "unknown"


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""

    # Check database connectivity
    try:
        db = get_database()
        await db.health_check()
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    return HealthResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        database=db_status,
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """Readiness check for Kubernetes."""

    # More thorough checks for readiness
    try:
        db = get_database()
        await db.health_check()

        return HealthResponse(
            status="ready",
            database="healthy",
        )
    except Exception as e:
        return HealthResponse(
            status="not_ready",
            database="unhealthy",
        )


@router.get("/live", response_model=HealthResponse)
async def liveness_check():
    """Liveness check for Kubernetes."""

    return HealthResponse(
        status="alive",
        database="unknown",  # Don't check DB for liveness
    )
