"""
Queue Service for Trace Ingestion

Handles enqueueing and dequeueing traces for async evaluation.
Supports Redis Streams and Kafka.
"""

import asyncio
import json
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog
from pydantic import BaseModel
from redis.asyncio import Redis

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: aiokafka not installed. Kafka support disabled.")

from metronis.core.models import Trace

logger = structlog.get_logger(__name__)


class QueueBackend(str, Enum):
    """Supported queue backends."""

    REDIS = "redis"
    KAFKA = "kafka"


class QueueService:
    """
    Service for enqueueing traces for async evaluation.

    Supports both Redis Streams and Apache Kafka.
    """

    def __init__(
        self,
        backend: QueueBackend = QueueBackend.REDIS,
        redis_url: Optional[str] = None,
        kafka_bootstrap_servers: Optional[str] = None,
        queue_name: str = "evaluation_queue",
    ):
        """
        Initialize queue service.

        Args:
            backend: Queue backend to use
            redis_url: Redis connection URL
            kafka_bootstrap_servers: Kafka bootstrap servers
            queue_name: Name of the queue/topic
        """
        self.backend = backend
        self.queue_name = queue_name

        # Initialize backend
        if backend == QueueBackend.REDIS:
            if not redis_url:
                redis_url = "redis://localhost:6379"
            self.redis_client = Redis.from_url(redis_url, decode_responses=True)
            self.kafka_producer = None
            self.kafka_consumer = None
        elif backend == QueueBackend.KAFKA:
            if not KAFKA_AVAILABLE:
                raise ValueError("Kafka support requires aiokafka package")
            if not kafka_bootstrap_servers:
                kafka_bootstrap_servers = "localhost:9092"
            self.kafka_bootstrap_servers = kafka_bootstrap_servers
            self.redis_client = None
            self.kafka_producer = None
            self.kafka_consumer = None
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        logger.info(
            "Queue service initialized",
            backend=backend,
            queue_name=queue_name,
        )

    async def enqueue(
        self,
        trace: Trace,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Enqueue a trace for evaluation.

        Args:
            trace: Trace to enqueue
            priority: Priority level (1-10, 10 = highest)
            metadata: Additional metadata

        Returns:
            Message ID
        """
        message = {
            "trace": trace.model_dump(mode="json"),
            "priority": priority,
            "metadata": metadata or {},
        }

        if self.backend == QueueBackend.REDIS:
            message_id = await self._enqueue_redis(message)
        elif self.backend == QueueBackend.KAFKA:
            message_id = await self._enqueue_kafka(message)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        logger.info(
            "Trace enqueued",
            trace_id=str(trace.trace_id),
            message_id=message_id,
            priority=priority,
        )

        return message_id

    async def _enqueue_redis(self, message: Dict[str, Any]) -> str:
        """Enqueue using Redis Streams."""
        # Add to Redis stream
        message_id = await self.redis_client.xadd(
            self.queue_name,
            {
                "data": json.dumps(message),
                "priority": message["priority"],
            },
        )

        # Add to priority-sorted set for priority queue behavior
        await self.redis_client.zadd(
            f"{self.queue_name}:priority",
            {message_id: message["priority"]},
        )

        return message_id

    async def _enqueue_kafka(self, message: Dict[str, Any]) -> str:
        """Enqueue using Kafka."""
        # Initialize producer if not already
        if self.kafka_producer is None:
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            await self.kafka_producer.start()

        # Send message
        metadata = await self.kafka_producer.send_and_wait(
            self.queue_name,
            value=message,
        )

        message_id = f"{metadata.topic}:{metadata.partition}:{metadata.offset}"
        return message_id

    async def dequeue(
        self,
        consumer_group: str = "default",
        consumer_name: str = "worker-1",
        block_ms: int = 1000,
    ) -> Optional[Dict[str, Any]]:
        """
        Dequeue a trace for evaluation.

        Args:
            consumer_group: Consumer group name
            consumer_name: Consumer name
            block_ms: Block time in milliseconds

        Returns:
            Message dict with trace and metadata, or None
        """
        if self.backend == QueueBackend.REDIS:
            return await self._dequeue_redis(consumer_group, consumer_name, block_ms)
        elif self.backend == QueueBackend.KAFKA:
            return await self._dequeue_kafka(consumer_group)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    async def _dequeue_redis(
        self, consumer_group: str, consumer_name: str, block_ms: int
    ) -> Optional[Dict[str, Any]]:
        """Dequeue using Redis Streams."""
        try:
            # Create consumer group if not exists
            try:
                await self.redis_client.xgroup_create(
                    self.queue_name, consumer_group, id="0", mkstream=True
                )
            except Exception:
                pass  # Group already exists

            # Read from stream
            messages = await self.redis_client.xreadgroup(
                groupname=consumer_group,
                consumername=consumer_name,
                streams={self.queue_name: ">"},
                count=1,
                block=block_ms,
            )

            if not messages:
                return None

            # Parse message
            stream_name, message_list = messages[0]
            message_id, message_data = message_list[0]

            # Parse JSON data
            data = json.loads(message_data["data"])

            # Reconstruct trace
            trace = Trace(**data["trace"])

            return {
                "message_id": message_id,
                "trace": trace,
                "priority": data.get("priority", 5),
                "metadata": data.get("metadata", {}),
            }

        except Exception as e:
            logger.error("Failed to dequeue from Redis", error=str(e))
            return None

    async def _dequeue_kafka(
        self, consumer_group: str
    ) -> Optional[Dict[str, Any]]:
        """Dequeue using Kafka."""
        # Initialize consumer if not already
        if self.kafka_consumer is None:
            self.kafka_consumer = AIOKafkaConsumer(
                self.queue_name,
                bootstrap_servers=self.kafka_bootstrap_servers,
                group_id=consumer_group,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
            )
            await self.kafka_consumer.start()

        try:
            # Get next message with timeout
            message = await asyncio.wait_for(
                self.kafka_consumer.getone(), timeout=1.0
            )

            # Parse message
            data = message.value

            # Reconstruct trace
            trace = Trace(**data["trace"])

            return {
                "message_id": f"{message.topic}:{message.partition}:{message.offset}",
                "trace": trace,
                "priority": data.get("priority", 5),
                "metadata": data.get("metadata", {}),
            }

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error("Failed to dequeue from Kafka", error=str(e))
            return None

    async def acknowledge(
        self,
        message_id: str,
        consumer_group: str = "default",
    ) -> None:
        """
        Acknowledge message processing.

        Args:
            message_id: Message ID to acknowledge
            consumer_group: Consumer group name
        """
        if self.backend == QueueBackend.REDIS:
            await self._acknowledge_redis(message_id, consumer_group)
        elif self.backend == QueueBackend.KAFKA:
            # Kafka auto-commits, but we can manually commit if needed
            if self.kafka_consumer:
                await self.kafka_consumer.commit()

        logger.debug("Message acknowledged", message_id=message_id)

    async def _acknowledge_redis(
        self, message_id: str, consumer_group: str
    ) -> None:
        """Acknowledge message in Redis."""
        await self.redis_client.xack(self.queue_name, consumer_group, message_id)

        # Remove from priority set
        await self.redis_client.zrem(f"{self.queue_name}:priority", message_id)

    async def get_queue_length(self) -> int:
        """Get current queue length."""
        if self.backend == QueueBackend.REDIS:
            return await self.redis_client.xlen(self.queue_name)
        elif self.backend == QueueBackend.KAFKA:
            # For Kafka, we'd need to check topic partitions
            # This is a simplified version
            return 0

    async def get_pending_messages(
        self, consumer_group: str = "default"
    ) -> int:
        """Get number of pending (unacknowledged) messages."""
        if self.backend == QueueBackend.REDIS:
            info = await self.redis_client.xpending(self.queue_name, consumer_group)
            return info["pending"]
        elif self.backend == QueueBackend.KAFKA:
            # Kafka doesn't have a direct equivalent
            return 0

    async def close(self) -> None:
        """Close connections."""
        if self.redis_client:
            await self.redis_client.close()

        if self.kafka_producer:
            await self.kafka_producer.stop()

        if self.kafka_consumer:
            await self.kafka_consumer.stop()

        logger.info("Queue service closed")


class PriorityQueueService(QueueService):
    """
    Enhanced queue service with priority handling.

    Always dequeues highest priority messages first.
    """

    async def dequeue(
        self,
        consumer_group: str = "default",
        consumer_name: str = "worker-1",
        block_ms: int = 1000,
    ) -> Optional[Dict[str, Any]]:
        """Dequeue highest priority message."""
        if self.backend != QueueBackend.REDIS:
            # Fall back to regular dequeue for non-Redis backends
            return await super().dequeue(consumer_group, consumer_name, block_ms)

        # Get highest priority message from sorted set
        priority_queue = f"{self.queue_name}:priority"

        # Get message IDs in priority order (highest first)
        message_ids = await self.redis_client.zrevrange(
            priority_queue, 0, 0, withscores=True
        )

        if not message_ids:
            # No messages, block and wait
            await asyncio.sleep(block_ms / 1000)
            return None

        message_id, priority = message_ids[0]

        # Read specific message
        messages = await self.redis_client.xread(
            {self.queue_name: message_id}, count=1
        )

        if not messages:
            return None

        # Parse message
        stream_name, message_list = messages[0]
        _, message_data = message_list[0]

        data = json.loads(message_data["data"])
        trace = Trace(**data["trace"])

        return {
            "message_id": message_id,
            "trace": trace,
            "priority": int(priority),
            "metadata": data.get("metadata", {}),
        }


# Global singleton
_queue_service: Optional[QueueService] = None


def get_queue_service(
    backend: QueueBackend = QueueBackend.REDIS,
    redis_url: Optional[str] = None,
    kafka_bootstrap_servers: Optional[str] = None,
) -> QueueService:
    """Get or create the global queue service instance."""
    global _queue_service
    if _queue_service is None:
        _queue_service = PriorityQueueService(
            backend=backend,
            redis_url=redis_url,
            kafka_bootstrap_servers=kafka_bootstrap_servers,
        )
    return _queue_service
