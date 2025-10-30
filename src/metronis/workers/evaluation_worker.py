"""
Evaluation Worker - COMPLETE P0 VERSION

Consumes traces from Redis queue and runs evaluation pipeline.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

import structlog

from metronis.core.domain import DomainRegistry
from metronis.core.models import EvaluationStatus, Trace
from metronis.core.orchestrator import FiveTierOrchestrator, ModuleRegistry
from metronis.db.session import SessionLocal
from metronis.infrastructure.repositories.evaluation_repository import (
    EvaluationRepository,
)
from metronis.infrastructure.repositories.trace_repository import TraceRepository
from metronis.workers.queue_service import QueueService

logger = structlog.get_logger(__name__)


class EvaluationWorker:
    """Worker that processes evaluation tasks from queue."""

    def __init__(
        self,
        domain_registry_path: Optional[Path] = None,
        redis_url: Optional[str] = None,
    ):
        # Initialize domain registry
        if domain_registry_path is None:
            domain_registry_path = Path(__file__).parent.parent.parent / "domains"

        self.domain_registry = DomainRegistry(domain_registry_path)
        logger.info("Loaded domains", domains=self.domain_registry.list_domains())

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

        logger.info("Evaluation worker initialized")

    def _register_modules(self) -> None:
        """Register evaluation modules from domains."""
        # Auto-registration happens in FiveTierOrchestrator
        logger.info("Modules registered", count=len(self.module_registry.modules))

    async def process_trace(self, trace: Trace) -> None:
        """Process a single trace through evaluation pipeline."""
        logger.info(
            "Starting trace evaluation",
            trace_id=str(trace.trace_id),
            domain=trace.metadata.domain if trace.metadata else "unknown",
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
                "Trace evaluation completed",
                trace_id=str(trace.trace_id),
                passed=evaluation_result.overall_passed,
                issues_count=len(evaluation_result.all_issues),
            )

        except Exception as e:
            logger.error(
                "Trace evaluation failed",
                trace_id=str(trace.trace_id),
                error=str(e),
                exc_info=True,
            )
        finally:
            db.close()

    async def run(self, queue_name: str = "evaluations") -> None:
        """Run worker loop, consuming from queue."""
        logger.info("Starting evaluation worker", queue=queue_name)

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
            logger.info("Worker shutting down")
        finally:
            self.queue.close()

    def get_queue_stats(self, queue_name: str = "evaluations") -> dict:
        """Get queue statistics."""
        return {
            "queue_length": self.queue.queue_length(queue_name),
            "queue_name": queue_name,
        }


async def main():
    """Main entry point for the worker."""
    worker = EvaluationWorker(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
