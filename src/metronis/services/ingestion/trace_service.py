"""Trace processing service."""

from typing import Optional
from uuid import UUID
import structlog

from metronis.core.models import Trace, EvaluationResult
from metronis.core.exceptions import ValidationError, TraceNotFoundError
from metronis.infrastructure.database import get_database
from metronis.infrastructure.repositories.trace_repository import TraceRepository
from metronis.infrastructure.repositories.evaluation_repository import EvaluationRepository
from metronis.infrastructure.sanitization import DataSanitizer
from metronis.infrastructure.queue import get_evaluation_queue


logger = structlog.get_logger(__name__)


class TraceService:
    """Service for processing traces."""
    
    def __init__(self):
        """Initialize trace service."""
        self.sanitizer = DataSanitizer()
        self.evaluation_queue = get_evaluation_queue()
    
    async def process_trace(self, trace: Trace) -> Trace:
        """
        Process a trace: validate, sanitize, store, and queue for evaluation.
        
        Args:
            trace: The trace to process
            
        Returns:
            The processed trace
            
        Raises:
            ValidationError: If trace validation fails
        """
        
        # Validate trace
        await self._validate_trace(trace)
        
        # Sanitize trace data
        sanitized_trace = self.sanitizer.sanitize_trace(trace)
        
        # Store trace in database
        db = get_database()
        trace_repo = TraceRepository(db.session)
        stored_trace = await trace_repo.create(sanitized_trace)
        
        # Queue for evaluation
        await self.evaluation_queue.enqueue_trace(stored_trace)
        
        logger.info(
            "Trace processed successfully",
            trace_id=str(stored_trace.trace_id),
            organization_id=str(stored_trace.organization_id),
        )
        
        return stored_trace
    
    async def get_trace(self, trace_id: str, organization_id: UUID) -> Optional[Trace]:
        """
        Get a trace by ID, ensuring it belongs to the organization.
        
        Args:
            trace_id: The trace ID
            organization_id: The organization ID
            
        Returns:
            The trace if found and authorized, None otherwise
        """
        
        db = get_database()
        trace_repo = TraceRepository(db.session)
        
        trace = await trace_repo.get_by_id(trace_id)
        
        if trace and trace.organization_id == organization_id:
            return trace
        
        return None
    
    async def get_evaluation_result(
        self, 
        trace_id: str, 
        organization_id: UUID
    ) -> Optional[EvaluationResult]:
        """
        Get evaluation result for a trace.
        
        Args:
            trace_id: The trace ID
            organization_id: The organization ID
            
        Returns:
            The evaluation result if found and authorized, None otherwise
        """
        
        # First verify the trace belongs to the organization
        trace = await self.get_trace(trace_id, organization_id)
        if not trace:
            return None
        
        # Get evaluation result
        db = get_database()
        eval_repo = EvaluationRepository(db.session)
        
        return await eval_repo.get_by_trace_id(trace_id)
    
    async def _validate_trace(self, trace: Trace) -> None:
        """
        Validate a trace.
        
        Args:
            trace: The trace to validate
            
        Raises:
            ValidationError: If validation fails
        """
        
        # Basic validation
        if not trace.ai_processing.input.strip():
            raise ValidationError("AI input cannot be empty")
        
        if not trace.ai_processing.output.strip():
            raise ValidationError("AI output cannot be empty")
        
        if not trace.ai_processing.model:
            raise ValidationError("AI model must be specified")
        
        # Size limits
        max_input_size = 100_000  # 100KB
        max_output_size = 100_000  # 100KB
        
        if len(trace.ai_processing.input) > max_input_size:
            raise ValidationError(f"Input size exceeds limit of {max_input_size} characters")
        
        if len(trace.ai_processing.output) > max_output_size:
            raise ValidationError(f"Output size exceeds limit of {max_output_size} characters")
        
        # Token validation
        if trace.ai_processing.tokens_used is not None:
            if trace.ai_processing.tokens_used < 0:
                raise ValidationError("Token count cannot be negative")
        
        # Latency validation
        if trace.ai_processing.latency_ms is not None:
            if trace.ai_processing.latency_ms < 0:
                raise ValidationError("Latency cannot be negative")
        
        logger.debug(
            "Trace validation passed",
            trace_id=str(trace.trace_id),
            input_length=len(trace.ai_processing.input),
            output_length=len(trace.ai_processing.output),
        )