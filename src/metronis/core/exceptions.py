"""Core exceptions for Metronis platform."""


class MetronisException(Exception):
    """Base exception for all Metronis errors."""

    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class ValidationError(MetronisException):
    """Raised when data validation fails."""

    pass


class AuthenticationError(MetronisException):
    """Raised when authentication fails."""

    pass


class AuthorizationError(MetronisException):
    """Raised when authorization fails."""

    pass


class RateLimitError(MetronisException):
    """Raised when rate limits are exceeded."""

    pass


class TraceNotFoundError(MetronisException):
    """Raised when a trace is not found."""

    pass


class EvaluationError(MetronisException):
    """Raised when evaluation fails."""

    pass


class ModuleError(MetronisException):
    """Raised when an evaluation module fails."""

    pass


class ConfigurationError(MetronisException):
    """Raised when configuration is invalid."""

    pass


class ExternalServiceError(MetronisException):
    """Raised when external service calls fail."""

    pass


class DatabaseError(MetronisException):
    """Raised when database operations fail."""

    pass
