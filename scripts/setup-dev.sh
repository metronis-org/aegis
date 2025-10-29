#!/bin/bash

# Metronis Development Setup Script

set -e

echo "🚀 Setting up Metronis development environment..."

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Please install Poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
poetry install

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please update .env file with your configuration"
fi

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
poetry run pre-commit install

# Start database and Redis services
echo "🐘 Starting PostgreSQL and Redis services..."
docker-compose up -d postgres redis

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Run database migrations
echo "🗄️  Running database migrations..."
poetry run alembic upgrade head

# Create initial organization (for development)
echo "🏢 Creating development organization..."
poetry run python scripts/create-dev-org.py

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Update .env file with your configuration"
echo "   2. Run 'make dev' to start the development server"
echo "   3. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "📚 Useful commands:"
echo "   make dev        - Start development server"
echo "   make test       - Run tests"
echo "   make lint       - Run linting"
echo "   make format     - Format code"
echo "   docker-compose logs -f  - View service logs"