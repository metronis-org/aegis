"""Queue service for worker to consume traces."""

import json
import os
from typing import Optional

import redis
import structlog

from metronis.core.models import Trace

logger = structlog.get_logger(__name__)


class QueueService:
    """Redis-based queue service for trace processing."""

    def __init__(self, redis_url: Optional[str] = None):
        if redis_url is None:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

        self.client = redis.from_url(redis_url, decode_responses=True)
        logger.info("Queue service initialized", redis_url=redis_url)

    async def enqueue_trace(
        self, trace: Trace, queue_name: str = "evaluations"
    ) -> None:
        """Add trace to evaluation queue."""
        trace_json = trace.model_dump_json()
        self.client.rpush(queue_name, trace_json)
        logger.debug("Trace enqueued", trace_id=str(trace.trace_id), queue=queue_name)

    async def dequeue_trace(
        self, queue_name: str = "evaluations", timeout: int = 1
    ) -> Optional[Trace]:
        """Get next trace from queue (blocking with timeout)."""
        result = self.client.blpop(queue_name, timeout=timeout)
        if result:
            _, trace_json = result
            trace_dict = json.loads(trace_json)
            return Trace(**trace_dict)
        return None

    def queue_length(self, queue_name: str = "evaluations") -> int:
        """Get current queue length."""
        return self.client.llen(queue_name)

    def close(self):
        """Close Redis connection."""
        self.client.close()
