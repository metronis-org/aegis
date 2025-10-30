"""Base repository with common functionality."""

from typing import Generic, List, Optional, Type, TypeVar
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from metronis.core.exceptions import DatabaseError

ModelType = TypeVar("ModelType", bound=DeclarativeBase)
SchemaType = TypeVar("SchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType, SchemaType]):
    """Base repository with common CRUD operations."""

    def __init__(self, session: AsyncSession, model: Type[ModelType]):
        """Initialize repository with database session and model."""
        self.session = session
        self.model = model

    async def create(self, obj: SchemaType) -> ModelType:
        """Create a new record."""
        try:
            # Convert Pydantic model to SQLAlchemy model
            db_obj = self.model(**obj.dict())

            self.session.add(db_obj)
            await self.session.flush()
            await self.session.refresh(db_obj)

            return db_obj

        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to create record: {str(e)}")

    async def get_by_id(self, id_value: UUID) -> Optional[ModelType]:
        """Get a record by ID."""
        try:
            # Assume the primary key is the first column
            pk_column = list(self.model.__table__.primary_key.columns)[0]

            stmt = select(self.model).where(pk_column == id_value)
            result = await self.session.execute(stmt)

            return result.scalar_one_or_none()

        except Exception as e:
            raise DatabaseError(f"Failed to get record by ID: {str(e)}")

    async def update(self, id_value: UUID, updates: dict) -> Optional[ModelType]:
        """Update a record by ID."""
        try:
            obj = await self.get_by_id(id_value)
            if not obj:
                return None

            for key, value in updates.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)

            await self.session.flush()
            await self.session.refresh(obj)

            return obj

        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to update record: {str(e)}")

    async def delete(self, id_value: UUID) -> bool:
        """Delete a record by ID."""
        try:
            obj = await self.get_by_id(id_value)
            if not obj:
                return False

            await self.session.delete(obj)
            await self.session.flush()

            return True

        except Exception as e:
            await self.session.rollback()
            raise DatabaseError(f"Failed to delete record: {str(e)}")

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[ModelType]:
        """List all records with pagination."""
        try:
            stmt = select(self.model).limit(limit).offset(offset)
            result = await self.session.execute(stmt)

            return result.scalars().all()

        except Exception as e:
            raise DatabaseError(f"Failed to list records: {str(e)}")

    async def count(self) -> int:
        """Count total records."""
        try:
            stmt = select(func.count()).select_from(self.model)
            result = await self.session.execute(stmt)

            return result.scalar()

        except Exception as e:
            raise DatabaseError(f"Failed to count records: {str(e)}")
