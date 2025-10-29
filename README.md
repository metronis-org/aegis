# Metronis - Unified AI Evaluation Platform

Metronis is a comprehensive AI evaluation platform designed to ensure the safety, accuracy, and reliability of AI systems across all domains. The platform employs a multi-tiered evaluation pipeline that balances cost-efficiency with accuracy, combining rule-based checks, ML classification, and LLM-as-judge evaluation.

## 🚀 Features

- **Multi-Tiered Evaluation**: Cost-efficient pipeline with Tier 1 (heuristics), Tier 2 (ML), and Tier 3 (LLM-as-judge)
- **Generic AI Support**: Evaluate any AI system - chatbots, RAG systems, agents, and more
- **Real-time Processing**: Handle 100K-10M+ traces per month with <5s evaluation latency
- **Modular Architecture**: Pluggable evaluation modules for easy extensibility
- **Production Ready**: Horizontal scaling, monitoring, and enterprise security
- **Developer Friendly**: Simple SDK integration with comprehensive documentation

aegis/
├── src/metronis/                    # Main source code
│   ├── core/                        # Core domain models & interfaces
│   │   ├── models.py               # Pydantic models (Trace, EvaluationResult, etc.)
│   │   ├── interfaces.py           # Abstract base classes & interfaces
│   │   ├── exceptions.py           # Custom exception classes
│   │   └── config.py               # Configuration management
│   ├── services/                    # Service layer
│   │   └── ingestion/              # Trace ingestion service
│   │       ├── app.py              # FastAPI application
│   │       ├── routes/             # API endpoints
│   │       ├── dependencies.py     # Dependency injection
│   │       └── trace_service.py    # Business logic
│   └── infrastructure/             # Infrastructure layer
│       ├── database.py             # Database connection management
│       ├── models/                 # SQLAlchemy database models
│       └── repositories/           # Data access layer
├── tests/                          # Comprehensive test suite
├── docker/                         # Docker configurations
├── migrations/                     # Database migrations (Alembic)
├── scripts/                        # Development scripts
└── config files                    # Poetry, Docker Compose, etc.


## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
│  AI Apps │ RAG Systems │ Chatbots │ Agents │ Other AI      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    API Gateway                              │
│        Authentication │ Rate Limiting │ Load Balancing      │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Core Services                                  │
│ Ingestion │ Orchestrator │ Evaluation │ Alerts │ Dashboard │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Data & Queue Layer                        │
│ PostgreSQL │ Redis │ Kafka │ Elasticsearch │ Monitoring    │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

- **Backend**: Python 3.11+, FastAPI, Celery
- **Database**: PostgreSQL, Redis
- **Queue**: Apache Kafka (Redis for development)
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker, Kubernetes
- **ML/AI**: Transformers, OpenAI/Anthropic APIs

## 📦 Quick Start

### Prerequisites

- Python 3.11+
- Poetry
- Docker & Docker Compose
- Git

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd aegis
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup-dev.sh
   ```

3. **Start the development server**
   ```bash
   make dev
   ```

4. **Visit the API documentation**
   - API Docs: http://localhost:8000/docs
   - Flower (Celery): http://localhost:5555
   - Grafana: http://localhost:3000 (admin/admin)

### Using the SDK

```python
from metronis import MetronisClient, Trace, AIProcessing

# Initialize client
client = MetronisClient(
    api_key="your-api-key",
    base_url="http://localhost:8000"
)

# Create a trace
trace = Trace(
    organization_id="your-org-id",
    application_id="your-app-id",
    ai_processing=AIProcessing(
        model="gpt-4",
        input="What is the capital of France?",
        output="The capital of France is Paris.",
    )
)

# Submit for evaluation
response = await client.trace(trace)
print(f"Trace ID: {response.trace_id}")

# Get evaluation result
result = await client.get_evaluation(response.trace_id)
print(f"Evaluation passed: {result.overall_passed}")
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-e2e

# Run with coverage
poetry run pytest --cov=src/metronis --cov-report=html
```

## 🔧 Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run all checks
make check
```

### Database Migrations

```bash
# Create new migration
make migrate-create

# Apply migrations
make migrate
```

### Docker Development

```bash
# Build images
make docker-build

# Start all services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

## 📊 Monitoring

The platform includes comprehensive monitoring:

- **Prometheus**: Metrics collection at http://localhost:9090
- **Grafana**: Dashboards at http://localhost:3000
- **Flower**: Celery monitoring at http://localhost:5555
- **Health Checks**: Available at `/health/`, `/health/ready`, `/health/live`

### Key Metrics

- Traces processed per second
- Evaluation latency (P95, P99)
- Error rates by tier and module
- Queue depths and processing times
- System resource utilization

## 🔒 Security

- **API Key Authentication**: Secure access control
- **Data Sanitization**: PII detection and masking
- **Encryption**: AES-256 at rest, TLS 1.3 in transit
- **Rate Limiting**: Configurable per organization
- **Audit Logging**: Comprehensive activity tracking

## 📈 Scaling

The platform is designed for horizontal scaling:

- **Stateless Services**: All services can be replicated
- **Auto-scaling**: Based on queue depth and resource usage
- **Database Scaling**: Read replicas and connection pooling
- **Caching**: Redis for frequently accessed data
- **Load Balancing**: Built-in support for multiple instances

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make check`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use type hints
- Follow conventional commit messages

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Architecture Guide](docs/architecture.md) - Detailed system design
- [Deployment Guide](docs/deployment.md) - Production deployment
- [SDK Reference](docs/sdk.md) - Client library documentation
- [Contributing Guide](CONTRIBUTING.md) - Development guidelines

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check if PostgreSQL is running
   docker-compose ps postgres
   
   # View database logs
   docker-compose logs postgres
   ```

2. **Redis Connection Errors**
   ```bash
   # Check Redis status
   docker-compose ps redis
   
   # Test Redis connection
   docker-compose exec redis redis-cli ping
   ```

3. **Import Errors**
   ```bash
   # Ensure dependencies are installed
   poetry install
   
   # Check Python path
   poetry run python -c "import metronis; print('OK')"
   ```

### Getting Help

- Check the [documentation](docs/)
- Search [existing issues](https://github.com/metronis/aegis/issues)
- Create a [new issue](https://github.com/metronis/aegis/issues/new)
- Join our [Discord community](https://discord.gg/metronis)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

**Metronis** - AI evals for everything AI 🛡️