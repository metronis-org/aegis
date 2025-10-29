# Metronis Development Makefile

.PHONY: help install dev test lint format clean docker-build docker-up docker-down migrate

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies with Poetry"
	@echo "  dev         - Start development environment"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting (flake8, mypy)"
	@echo "  format      - Format code (black, isort)"
	@echo "  clean       - Clean up temporary files"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up   - Start Docker services"
	@echo "  docker-down - Stop Docker services"
	@echo "  migrate     - Run database migrations"

# Install dependencies
install:
	poetry install

# Start development environment
dev:
	docker-compose up -d postgres redis
	@echo "Waiting for services to be ready..."
	@sleep 5
	poetry run uvicorn metronis.services.ingestion.app:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	poetry run pytest -v --cov=src/metronis --cov-report=term-missing

# Run linting
lint:
	poetry run flake8 src/ tests/
	poetry run mypy src/

# Format code
format:
	poetry run black src/ tests/
	poetry run isort src/ tests/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Database migrations
migrate:
	poetry run alembic upgrade head

migrate-create:
	@read -p "Enter migration message: " msg; \
	poetry run alembic revision --autogenerate -m "$$msg"

# Development setup
setup-dev: install
	poetry run pre-commit install
	@echo "Development environment setup complete!"

# Run specific service
run-ingestion:
	poetry run uvicorn metronis.services.ingestion.app:app --reload --host 0.0.0.0 --port 8000

run-worker:
	poetry run celery -A metronis.services.evaluation.worker worker --loglevel=info

run-flower:
	poetry run celery -A metronis.services.evaluation.worker flower --port=5555

# Testing commands
test-unit:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

test-e2e:
	poetry run pytest tests/e2e/ -v

# Code quality
check: lint test
	@echo "All checks passed!"

# Build and push Docker images (for CI/CD)
docker-push:
	docker tag metronis-ingestion:latest metronis/ingestion:$(VERSION)
	docker tag metronis-worker:latest metronis/worker:$(VERSION)
	docker push metronis/ingestion:$(VERSION)
	docker push metronis/worker:$(VERSION)