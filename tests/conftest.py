"""Pytest configuration and fixtures."""

import asyncio
import pytest
from typing import AsyncGenerator
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from testcontainers.postgres import PostgresContainer

from metronis.core.models import Organization, Trace, AIProcessing
from metronis.infrastructure.database import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def postgres_container():
    """Start PostgreSQL container for testing."""
    with PostgresContainer("postgres:15-alpine") as postgres:
        yield postgres


@pytest.fixture(scope="session")
async def test_engine(postgres_container):
    """Create test database engine."""
    database_url = postgres_container.get_connection_url().replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    
    engine = create_async_engine(database_url, echo=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with session_factory() as session:
        try:
            yield session
            await session.rollback()
        finally:
            await session.close()


@pytest.fixture
def sample_organization() -> Organization:
    """Create a sample organization for testing."""
    return Organization(
        organization_id=uuid4(),
        name="Test Organization",
        api_key_hash="test_api_key_hash",
        rate_limit_per_minute=1000,
    )


@pytest.fixture
def sample_trace(sample_organization: Organization) -> Trace:
    """Create a sample trace for testing."""
    return Trace(
        organization_id=sample_organization.organization_id,
        application_id=uuid4(),
        ai_processing=AIProcessing(
            model="gpt-4",
            input="What is the capital of France?",
            output="The capital of France is Paris.",
            tokens_used=50,
            latency_ms=1200,
        ),
    )


@pytest.fixture
def sample_traces(sample_organization: Organization) -> list[Trace]:
    """Create multiple sample traces for testing."""
    traces = []
    for i in range(5):
        trace = Trace(
            organization_id=sample_organization.organization_id,
            application_id=uuid4(),
            ai_processing=AIProcessing(
                model=f"model-{i}",
                input=f"Test input {i}",
                output=f"Test output {i}",
                tokens_used=50 + i * 10,
                latency_ms=1000 + i * 100,
            ),
        )
        traces.append(trace)
    return traces