# Metronis Aegis - P0 Deployment Guide

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
docker-compose exec postgres psql -U metronis -d metronis -c "\dt"
```

### Step 3: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Expected response: {"status":"healthy"}

# Create a test organization (get API key)
curl -X POST http://localhost:8000/api/v1/organizations \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Org"}'

# Submit a trace for evaluation
curl -X POST http://localhost:8000/api/v1/traces \
  -H "Authorization: Bearer YOUR_API_KEY_HERE" \
  -H "Content-Type: application/json" \
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
metronis=# \dt

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
