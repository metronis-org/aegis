"""
Complete P0 Implementation Script
Generates all remaining P0 files to make the system fully operational.
"""

import os
from pathlib import Path


def create_file(path: str, content: str):
    """Create a file with the given content."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    print(f"Created {path}")


# =============================================================================
# 1. Complete Worker Queue Consumption
# =============================================================================

worker_queue_service = """'''Queue service for worker to consume traces.'''

import json
import os
from typing import Optional
import redis
import structlog
from metronis.core.models import Trace

logger = structlog.get_logger(__name__)


class QueueService:
    '''Redis-based queue service for trace processing.'''

    def __init__(self, redis_url: Optional[str] = None):
        if redis_url is None:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

        self.client = redis.from_url(redis_url, decode_responses=True)
        logger.info('Queue service initialized', redis_url=redis_url)

    async def enqueue_trace(self, trace: Trace, queue_name: str = 'evaluations') -> None:
        '''Add trace to evaluation queue.'''
        trace_json = trace.model_dump_json()
        self.client.rpush(queue_name, trace_json)
        logger.debug('Trace enqueued', trace_id=str(trace.trace_id), queue=queue_name)

    async def dequeue_trace(self, queue_name: str = 'evaluations', timeout: int = 1) -> Optional[Trace]:
        '''Get next trace from queue (blocking with timeout).'''
        result = self.client.blpop(queue_name, timeout=timeout)
        if result:
            _, trace_json = result
            trace_dict = json.loads(trace_json)
            return Trace(**trace_dict)
        return None

    def queue_length(self, queue_name: str = 'evaluations') -> int:
        '''Get current queue length.'''
        return self.client.llen(queue_name)

    def close(self):
        '''Close Redis connection.'''
        self.client.close()
"""

# Updated worker with queue consumption
complete_worker = """'''
Evaluation Worker - COMPLETE P0 VERSION

Consumes traces from Redis queue and runs evaluation pipeline.
'''

import asyncio
import os
from pathlib import Path
from typing import Optional

import structlog

from metronis.core.domain import DomainRegistry
from metronis.core.orchestrator import FiveTierOrchestrator, ModuleRegistry
from metronis.core.models import Trace, EvaluationStatus
from metronis.db.session import SessionLocal
from metronis.infrastructure.repositories.trace_repository import TraceRepository
from metronis.infrastructure.repositories.evaluation_repository import EvaluationRepository
from metronis.workers.queue_service import QueueService

logger = structlog.get_logger(__name__)


class EvaluationWorker:
    '''Worker that processes evaluation tasks from queue.'''

    def __init__(
        self,
        domain_registry_path: Optional[Path] = None,
        redis_url: Optional[str] = None,
    ):
        # Initialize domain registry
        if domain_registry_path is None:
            domain_registry_path = Path(__file__).parent.parent.parent / 'domains'

        self.domain_registry = DomainRegistry(domain_registry_path)
        logger.info('Loaded domains', domains=self.domain_registry.list_domains())

        # Initialize module registry
        self.module_registry = ModuleRegistry()
        self._register_modules()

        # Initialize queue service
        self.queue = QueueService(redis_url=redis_url)

        # Initialize orchestrator
        self.orchestrator = FiveTierOrchestrator(
            domain_registry=self.domain_registry,
            module_registry=self.module_registry,
        )

        logger.info('Evaluation worker initialized')

    def _register_modules(self) -> None:
        '''Register evaluation modules from domains.'''
        # Auto-registration happens in FiveTierOrchestrator
        logger.info('Modules registered', count=len(self.module_registry.modules))

    async def process_trace(self, trace: Trace) -> None:
        '''Process a single trace through evaluation pipeline.'''
        logger.info(
            'Starting trace evaluation',
            trace_id=str(trace.trace_id),
            domain=trace.metadata.domain if trace.metadata else 'unknown',
        )

        db = SessionLocal()
        try:
            # Update status to in_progress
            trace_repo = TraceRepository(db)
            # trace_repo.update_status(trace.trace_id, 'in_progress')

            # Run evaluation
            evaluation_result = await self.orchestrator.evaluate_trace(trace)

            # Store evaluation result
            eval_repo = EvaluationRepository(db)
            eval_repo.create(evaluation_result)

            logger.info(
                'Trace evaluation completed',
                trace_id=str(trace.trace_id),
                passed=evaluation_result.overall_passed,
                issues_count=len(evaluation_result.all_issues),
            )

        except Exception as e:
            logger.error(
                'Trace evaluation failed',
                trace_id=str(trace.trace_id),
                error=str(e),
                exc_info=True,
            )
        finally:
            db.close()

    async def run(self, queue_name: str = 'evaluations') -> None:
        '''Run worker loop, consuming from queue.'''
        logger.info('Starting evaluation worker', queue=queue_name)

        try:
            while True:
                # Dequeue next trace (blocks for 1 second)
                trace = await self.queue.dequeue_trace(queue_name)

                if trace:
                    await self.process_trace(trace)
                else:
                    # No work available, short sleep
                    await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info('Worker shutting down')
        finally:
            self.queue.close()

    def get_queue_stats(self, queue_name: str = 'evaluations') -> dict:
        '''Get queue statistics.'''
        return {
            'queue_length': self.queue.queue_length(queue_name),
            'queue_name': queue_name,
        }


async def main():
    '''Main entry point for the worker.'''
    worker = EvaluationWorker(
        redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    )
    await worker.run()


if __name__ == '__main__':
    asyncio.run(main())
"""

# =============================================================================
# 2. Docker Images
# =============================================================================

dockerfile_api = """# Dockerfile for Metronis Aegis API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    postgresql-client \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN pip install --no-cache-dir poetry && \\
    poetry config virtualenvs.create false && \\
    poetry install --no-dev --no-interaction --no-ansi

# Copy source code
COPY src/ /app/src/
COPY alembic/ /app/alembic/
COPY alembic.ini /app/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run API
CMD ["python", "-m", "uvicorn", "metronis.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

dockerfile_worker = """# Dockerfile for Metronis Aegis Worker
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    postgresql-client \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN pip install --no-cache-dir poetry && \\
    poetry config virtualenvs.create false && \\
    poetry install --no-dev --no-interaction --no-ansi

# Copy source code
COPY src/ /app/src/
COPY domains/ /app/domains/

# Set Python path
ENV PYTHONPATH=/app/src

# Run worker
CMD ["python", "-m", "metronis.workers.evaluation_worker"]
"""

# =============================================================================
# 3. Database initialization script
# =============================================================================

init_db_script = """#!/usr/bin/env python
'''Initialize database with migrations.'''

import subprocess
import sys
import os

def main():
    print('[INFO] Running database migrations...')

    # Run alembic upgrade
    result = subprocess.run(
        ['python', '-m', 'alembic', 'upgrade', 'head'],
        env={**os.environ, 'PYTHONPATH': 'src'},
    )

    if result.returncode == 0:
        print('[OK] Database initialized successfully!')
    else:
        print('[ERROR] Database initialization failed')
        sys.exit(1)

if __name__ == '__main__':
    main()
"""

# =============================================================================
# 4. Updated docker-compose for P0
# =============================================================================

docker_compose_p0 = """version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: metronis
      POSTGRES_USER: metronis
      POSTGRES_PASSWORD: metronis_dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U metronis -d metronis"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis for queue and caching
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # API Service
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://metronis:metronis_dev_password@postgres:5432/metronis
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./src:/app/src
      - ./domains:/app/domains
    command: uvicorn metronis.api.main:app --host 0.0.0.0 --port 8000 --reload

  # Evaluation Worker
  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    environment:
      - DATABASE_URL=postgresql://metronis:metronis_dev_password@postgres:5432/metronis
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./src:/app/src
      - ./domains:/app/domains

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    name: metronis-network
"""

# =============================================================================
# 5. P0 Deployment README
# =============================================================================

deployment_readme = """# Metronis Aegis - P0 Deployment Guide

## What is P0?

P0 is the **minimum viable production infrastructure** for Metronis Aegis. It includes:

- ✅ Complete database layer (PostgreSQL with SQLAlchemy ORM)
- ✅ REST API with authentication (FastAPI)
- ✅ Worker queue for async evaluation (Redis queue)
- ✅ 5-Tier evaluation orchestrator
- ✅ Domain-specific evaluation system
- ✅ Docker deployment

## Prerequisites

To run Metronis Aegis P0, you need:

### 1. Software Requirements
- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
  - Download: https://www.docker.com/products/docker-desktop/
  - Version: 20.10+ recommended

- **Docker Compose** (usually included with Docker Desktop)
  - Version: 2.0+ recommended

### 2. Hardware Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 20GB free space
- **CPU**: 2 cores minimum, 4 cores recommended

### 3. API Keys (Optional for full features)
- **OpenAI API Key** (for Tier-3 LLM evaluations)
  - Get one at: https://platform.openai.com/api-keys
  - Set as environment variable: `OPENAI_API_KEY`

- **Anthropic API Key** (for Tier-3 LLM evaluations)
  - Get one at: https://console.anthropic.com/
  - Set as environment variable: `ANTHROPIC_API_KEY`

## Quick Start (3 Steps)

### Step 1: Start Services

```bash
# Start PostgreSQL, Redis, API, and Worker
docker-compose up -d

# Wait for services to be healthy (takes ~30 seconds)
docker-compose ps
```

### Step 2: Initialize Database

```bash
# Run database migrations (creates all tables)
docker-compose exec api python scripts/init_db.py

# Verify tables were created
docker-compose exec postgres psql -U metronis -d metronis -c "\\dt"
```

### Step 3: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Expected response: {"status":"healthy"}

# Create a test organization (get API key)
curl -X POST http://localhost:8000/api/v1/organizations \\
  -H "Content-Type: application/json" \\
  -d '{"name": "Test Org"}'

# Submit a trace for evaluation
curl -X POST http://localhost:8000/api/v1/traces \\
  -H "Authorization: Bearer YOUR_API_KEY_HERE" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "gpt-4",
    "input": "What is the capital of France?",
    "output": "The capital of France is Paris.",
    "domain": "healthcare"
  }'
```

## System Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────>│  FastAPI     │─────>│   Redis     │
│   (curl)    │      │  (Port 8000) │      │   Queue     │
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            ▼                      ▼
                     ┌──────────────┐      ┌─────────────┐
                     │  PostgreSQL  │<─────│   Worker    │
                     │  (Port 5432) │      │  (Async)    │
                     └──────────────┘      └─────────────┘
```

## Service Endpoints

| Service    | Port | URL                        | Purpose                    |
|------------|------|----------------------------|----------------------------|
| API        | 8000 | http://localhost:8000      | REST API                   |
| PostgreSQL | 5432 | localhost:5432             | Database                   |
| Redis      | 6379 | localhost:6379             | Queue & Cache              |

## Environment Variables

Create a `.env` file in the root directory:

```bash
# Database
DATABASE_URL=postgresql://metronis:metronis_dev_password@postgres:5432/metronis

# Redis
REDIS_URL=redis://redis:6379/0

# API Keys (optional)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Logging
LOG_LEVEL=INFO
```

## Development Mode

For local development without Docker:

```bash
# 1. Install dependencies
pip install -e .

# 2. Start PostgreSQL and Redis (using Docker)
docker-compose up -d postgres redis

# 3. Run migrations
python -m alembic upgrade head

# 4. Start API (with hot reload)
python -m uvicorn metronis.api.main:app --reload

# 5. Start worker (in another terminal)
python -m metronis.workers.evaluation_worker
```

## Monitoring

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f worker
```

### Check Queue Status
```bash
# Connect to Redis
docker-compose exec redis redis-cli

# Check queue length
127.0.0.1:6379> LLEN evaluations

# View queue contents
127.0.0.1:6379> LRANGE evaluations 0 -1
```

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U metronis -d metronis

# List tables
metronis=# \\dt

# Query traces
metronis=# SELECT trace_id, model, domain FROM traces LIMIT 5;
```

## Troubleshooting

### Issue: "Docker daemon is not running"
**Solution**: Start Docker Desktop

### Issue: "Port 5432 already in use"
**Solution**: Stop local PostgreSQL or change port in docker-compose.yml

### Issue: "Migration failed"
**Solution**: Drop database and recreate:
```bash
docker-compose down -v
docker-compose up -d postgres redis
docker-compose exec postgres psql -U metronis -c "DROP DATABASE metronis;"
docker-compose exec postgres psql -U metronis -c "CREATE DATABASE metronis;"
python -m alembic upgrade head
```

### Issue: "Worker not processing traces"
**Solution**: Check worker logs:
```bash
docker-compose logs -f worker
# Ensure Redis is healthy
docker-compose ps redis
```

## What's Next? (P1 Features)

After P0 is working, these are the next priority features:

1. **Frontend Dashboard** - React UI for viewing traces and evaluations
2. **Billing System** - Stripe integration for usage-based billing
3. **Compliance Reports** - Auto-generated HIPAA/SOC2 documentation
4. **More Domains** - Add trading, robotics, legal evaluators
5. **Active Learning UI** - Interface for expert labeling

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: https://github.com/metronis-org/aegis/issues
- **Email**: support@metronis.com

## Production Deployment

For production deployment (AWS/GCP/Azure):

1. Use managed PostgreSQL (RDS, Cloud SQL)
2. Use managed Redis (ElastiCache, Memorystore)
3. Deploy API/Worker to Kubernetes (see `k8s/` directory)
4. Set up monitoring (Prometheus, Grafana)
5. Configure SSL/TLS certificates
6. Set up CI/CD pipeline (see `.github/workflows/`)

See `docs/production-deployment.md` for full guide.
"""

# =============================================================================
# Execute: Create all files
# =============================================================================

files_to_create = {
    "src/metronis/workers/queue_service.py": worker_queue_service,
    "src/metronis/workers/evaluation_worker.py": complete_worker,
    "docker/Dockerfile.api": dockerfile_api,
    "docker/Dockerfile.worker": dockerfile_worker,
    "scripts/init_db.py": init_db_script,
    "docker-compose.p0.yml": docker_compose_p0,
    "P0_DEPLOYMENT.md": deployment_readme,
}

if __name__ == "__main__":
    print("Creating P0 completion files...")
    print("=" * 60)

    for file_path, content in files_to_create.items():
        create_file(file_path, content)

    print("=" * 60)
    print(f"\\nCreated {len(files_to_create)} files!")
    print("\\nP0 IMPLEMENTATION COMPLETE!")
    print("\\nNext steps:")
    print("1. Read P0_DEPLOYMENT.md for prerequisites")
    print("2. Install Docker Desktop if not already installed")
    print("3. Run: docker-compose -f docker-compose.p0.yml up -d")
    print("4. Run: docker-compose exec api python scripts/init_db.py")
    print("5. Test: curl http://localhost:8000/health")
