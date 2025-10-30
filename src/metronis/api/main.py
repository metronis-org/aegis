"""FastAPI main application."""

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from metronis.api import analytics
from metronis.api.routes import (  # P2
    billing,
    compliance,
    evaluations,
    expert_review,
    onboarding,
    search,
    traces,
    websocket,
)

logger = structlog.get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Metronis Aegis API",
    description="Domain-specific, RL-native AI evaluation platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - load from config in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # TODO: Load from metronis.config.settings.security.cors_origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers - P0
app.include_router(traces.router, prefix="/api/v1", tags=["traces"])
app.include_router(evaluations.router, prefix="/api/v1", tags=["evaluations"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])

# Include routers - P1
app.include_router(billing.router, prefix="/api/v1", tags=["billing"])
app.include_router(compliance.router, prefix="/api/v1", tags=["compliance"])
app.include_router(onboarding.router, prefix="/api/v1", tags=["onboarding"])
app.include_router(websocket.router, tags=["websocket"])

# Include routers - P2 (NEW)
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(expert_review.router, prefix="/api/v1", tags=["expert-review"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "metronis-api"}


@app.get("/health/ready")
async def readiness():
    """Readiness check endpoint."""
    # TODO: Add database connectivity check
    return {"status": "ready"}


@app.on_event("startup")
async def startup_event():
    """Application startup."""
    logger.info("Metronis API starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown."""
    logger.info("Metronis API shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
