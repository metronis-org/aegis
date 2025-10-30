"""
Centralized Configuration Management

Environment-aware configuration using Pydantic BaseSettings.
"""

import os
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    url: str = Field(
        default="postgresql://metronis:metronis_dev_password@localhost:5432/metronis",
        env="DATABASE_URL",
    )
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=40, env="DB_MAX_OVERFLOW")
    echo: bool = Field(default=False, env="DB_ECHO")


class RedisSettings(BaseSettings):
    """Redis configuration."""

    url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")


class StripeSettings(BaseSettings):
    """Stripe billing configuration."""

    secret_key: str = Field(default="sk_test_...", env="STRIPE_SECRET_KEY")
    publishable_key: str = Field(default="pk_test_...", env="STRIPE_PUBLISHABLE_KEY")
    webhook_secret: str = Field(default="whsec_...", env="STRIPE_WEBHOOK_SECRET")


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(
        default=None, env="AZURE_OPENAI_ENDPOINT"
    )
    azure_openai_key: Optional[str] = Field(default=None, env="AZURE_OPENAI_KEY")


class SecuritySettings(BaseSettings):
    """Security configuration."""

    api_key_length: int = Field(default=32, env="API_KEY_LENGTH")
    jwt_secret_key: str = Field(default="change-me-in-production", env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS",
    )


class Settings(BaseSettings):
    """Main application settings."""

    # Application
    app_name: str = Field(default="Metronis Aegis", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    stripe: StripeSettings = StripeSettings()
    llm: LLMSettings = LLMSettings()
    security: SecuritySettings = SecuritySettings()

    # Worker
    worker_concurrency: int = Field(default=4, env="WORKER_CONCURRENCY")
    queue_name: str = Field(default="evaluations", env="QUEUE_NAME")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    enable_tracing: bool = Field(default=True, env="ENABLE_TRACING")

    @field_validator("environment")
    def validate_environment(cls, v):
        """Validate environment is one of: development, staging, production."""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
