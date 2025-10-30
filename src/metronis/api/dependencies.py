"""FastAPI dependencies."""

from typing import Generator

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from metronis.db.models import OrganizationModel
from metronis.db.session import get_db
from metronis.infrastructure.repositories.organization_repository import (
    OrganizationRepository,
)

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db),
) -> OrganizationModel:
    """
    Get current authenticated user/organization from API key.

    Usage:
        @router.get("/items")
        async def get_items(current_user = Depends(get_current_user)):
            return {"org": current_user.name}
    """
    api_key = credentials.credentials

    org_repo = OrganizationRepository(db)
    organization = org_repo.get_by_api_key(api_key)

    if not organization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    if not organization.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization is inactive",
        )

    return organization
