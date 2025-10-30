"""FastAPI application for trace ingestion service."""

from contextlib import asynccontextmanager
from typing import List

import structlog
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from metronis.core.config import settings
from metronis.core.exceptions import MetronisException
from metronis.core.models import EvaluationResult, Trace
from metronis.infrastructure.database import get_database
from metronis.infrastructure.monitoring import setup_monitoring
from metronis.services.ingestion.routes import health, traces

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting trace ingestion service")

    # Initialize database
    await get_database().initialize()

    # Setup monitoring
    setup_monitoring(app)

    yield

    # Shutdown
    logger.info("Shutting down trace ingestion service")
    await get_database().close()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="Metronis Trace Ingestion Service",
        description="Service for ingesting and validating AI traces",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.api.debug else None,
        redoc_url="/redoc" if settings.api.debug else None,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handler
    @app.exception_handler(MetronisException)
    async def metronis_exception_handler(request, exc: MetronisException):
        """Handle Metronis-specific exceptions."""
        return JSONResponse(
            status_code=400,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        """Handle general exceptions."""
        logger.error("Unhandled exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An internal server error occurred",
            },
        )

    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(traces.router, prefix="/v1", tags=["traces"])

    return app


# Create app instance
app = create_app()
