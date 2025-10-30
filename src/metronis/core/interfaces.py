"""Core interfaces and abstract base classes for Metronis."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from metronis.core.models import EvaluationResult, ModuleResult, Trace


class EvaluationModule(ABC):
    """Abstract base class for all evaluation modules."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evaluation module with configuration."""
        self.config = config or {}
        self.name = self.__class__.__name__
        self.tier_level = self.get_tier_level()

    @abstractmethod
    def get_tier_level(self) -> int:
        """Return the tier level (1, 2, or 3) for this module."""
        pass

    @abstractmethod
    def evaluate(
        self, trace: Trace, context: Optional[Dict[str, Any]] = None
    ) -> ModuleResult:
        """
        Evaluate a trace and return the result.

        Args:
            trace: The trace to evaluate
            context: Optional context from previous evaluation tiers

        Returns:
            ModuleResult containing the evaluation outcome
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about this module."""
        return {
            "name": self.name,
            "tier_level": self.tier_level,
            "description": self.__doc__ or "",
            "config": self.config,
        }

    def is_applicable(self, trace: Trace) -> bool:
        """
        Check if this module is applicable to the given trace.

        Override this method to implement application-type specific logic.
        """
        return True


class TraceRepository(ABC):
    """Abstract repository interface for trace storage."""

    @abstractmethod
    async def create(self, trace: Trace) -> Trace:
        """Create a new trace."""
        pass

    @abstractmethod
    async def get_by_id(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        pass

    @abstractmethod
    async def update(self, trace: Trace) -> Trace:
        """Update an existing trace."""
        pass

    @abstractmethod
    async def list_by_organization(
        self, organization_id: str, limit: int = 100, offset: int = 0
    ) -> List[Trace]:
        """List traces for an organization."""
        pass


class EvaluationRepository(ABC):
    """Abstract repository interface for evaluation results."""

    @abstractmethod
    async def create(self, result: EvaluationResult) -> EvaluationResult:
        """Create a new evaluation result."""
        pass

    @abstractmethod
    async def get_by_trace_id(self, trace_id: str) -> Optional[EvaluationResult]:
        """Get evaluation result by trace ID."""
        pass

    @abstractmethod
    async def update(self, result: EvaluationResult) -> EvaluationResult:
        """Update an existing evaluation result."""
        pass


class ModuleRegistry(ABC):
    """Abstract registry for evaluation modules."""

    @abstractmethod
    def register(self, module: EvaluationModule) -> None:
        """Register an evaluation module."""
        pass

    @abstractmethod
    def get_modules_for_tier(self, tier: int) -> List[EvaluationModule]:
        """Get all modules for a specific tier."""
        pass

    @abstractmethod
    def get_applicable_modules(self, trace: Trace, tier: int) -> List[EvaluationModule]:
        """Get modules applicable to a trace for a specific tier."""
        pass


class AlertService(ABC):
    """Abstract interface for alert services."""

    @abstractmethod
    async def send_alert(
        self,
        trace: Trace,
        evaluation_result: EvaluationResult,
        channels: Optional[List[str]] = None,
    ) -> None:
        """Send an alert for a critical evaluation result."""
        pass


class EvaluationOrchestrator(ABC):
    """Abstract interface for evaluation orchestration."""

    @abstractmethod
    async def evaluate_trace(self, trace: Trace) -> EvaluationResult:
        """Orchestrate the complete evaluation of a trace."""
        pass


class DataSanitizer(ABC):
    """Abstract interface for data sanitization."""

    @abstractmethod
    def sanitize_trace(self, trace: Trace) -> Trace:
        """Sanitize a trace by removing/masking sensitive information."""
        pass

    @abstractmethod
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text and return detection results."""
        pass
