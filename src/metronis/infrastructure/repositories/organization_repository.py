"""Organization repository."""

from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from metronis.db.models import OrganizationModel


class OrganizationRepository:
    """Repository for organizations."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, organization_id: UUID) -> Optional[OrganizationModel]:
        """Get organization by ID."""
        return (
            self.db.query(OrganizationModel)
            .filter(OrganizationModel.organization_id == organization_id)
            .first()
        )

    def get_by_api_key(self, api_key: str) -> Optional[OrganizationModel]:
        """Get organization by API key."""
        return (
            self.db.query(OrganizationModel)
            .filter(OrganizationModel.api_key == api_key)
            .first()
        )

    def create(
        self, name: str, api_key: str, domain: Optional[str] = None
    ) -> OrganizationModel:
        """Create new organization."""
        org = OrganizationModel(
            name=name,
            api_key=api_key,
            domain=domain,
        )
        self.db.add(org)
        self.db.commit()
        self.db.refresh(org)
        return org
