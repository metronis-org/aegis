"""Dependencies for ingestion service."""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from metronis.core.exceptions import AuthenticationError, AuthorizationError
from metronis.core.models import Organization
from metronis.infrastructure.database import get_database
from metronis.infrastructure.repositories.organization_repository import (
    OrganizationRepository,
)
from metronis.services.ingestion.trace_service import TraceService

logger = structlog.get_logger(__name__)
security = HTTPBearer()


async def get_organization_repository() -> OrganizationRepository:
    """Get organization repository instance."""
    db = get_database()
    return OrganizationRepository(db.session)


async def get_trace_service() -> TraceService:
    """Get trace service instance."""
    return TraceService()


async def get_current_organization(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    org_repo: OrganizationRepository = Depends(get_organization_repository),
) -> Organization:
    """Get current organization from API key."""

    try:
        api_key = credentials.credentials

        # Look up organization by API key
        organization = await org_repo.get_by_api_key(api_key)

        if not organization:
            logger.warning("Invalid API key used", api_key_prefix=api_key[:8])
            raise AuthenticationError("Invalid API key")

        logger.debug(
            "Organization authenticated",
            organization_id=str(organization.organization_id),
            organization_name=organization.name,
        )

        return organization

    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed",
        )
