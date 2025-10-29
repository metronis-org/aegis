"""
Metronis SDK Client Library

Simple Python SDK for integrating Metronis evaluation into your AI applications.

Usage:
    ```python
    from metronis.sdk import MetronisClient

    client = MetronisClient(api_key="your-key")

    # Evaluate an LLM trace
    result = await client.evaluate_trace({
        "input": "What medication for hypertension?",
        "output": "Consider lisinopril 10mg daily",
        "model": "gpt-4",
        "domain": "healthcare"
    })

    print(f"Safe: {result.overall_passed}")
    ```
"""

import asyncio
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import httpx

from metronis.core.models import (
    AIProcessing,
    ApplicationType,
    EvaluationResult,
    PolicyInfo,
    RLStep,
    Trace,
    TraceMetadata,
    UserContext,
)


class MetronisClient:
    """
    Main client for interacting with Metronis API.

    Handles:
    - Trace submission
    - Evaluation retrieval
    - Async and sync interfaces
    - Retries and error handling
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        organization_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the Metronis client.

        Args:
            api_key: Your Metronis API key
            base_url: URL of Metronis API (default: localhost)
            organization_id: Your organization ID (auto-detected if not provided)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.organization_id = organization_id
        self.timeout = timeout

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

        # Sync client for sync methods
        self._sync_client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def evaluate_trace(
        self,
        input: str,
        output: str,
        model: str,
        domain: Optional[str] = None,
        application_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        rl_episode: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> EvaluationResult:
        """
        Evaluate a single trace.

        Args:
            input: Input to the AI system
            output: Output from the AI system
            model: Model name (e.g., "gpt-4")
            domain: Domain name (healthcare, trading, robotics, etc.)
            application_type: Application type (clinical_support, rag, etc.)
            metadata: Additional metadata
            rl_episode: For RL agents, list of (state, action, reward) dicts
            **kwargs: Additional fields for Trace

        Returns:
            EvaluationResult with evaluation details
        """
        # Build trace
        trace = Trace(
            trace_id=uuid4(),
            organization_id=UUID(self.organization_id) if self.organization_id else uuid4(),
            application_id=kwargs.get("application_id", uuid4()),
            application_type=ApplicationType(application_type) if application_type else ApplicationType.GENERIC,
            ai_processing=AIProcessing(
                model=model,
                input=input,
                output=output,
                rl_episode=[RLStep(**step) for step in rl_episode] if rl_episode else [],
                **{k: v for k, v in kwargs.items() if k in AIProcessing.__fields__},
            ),
            metadata=TraceMetadata(
                domain=domain,
                **(metadata or {}),
            ),
        )

        # Submit trace
        response = await self._client.post(
            "/api/v1/traces",
            json=trace.model_dump(mode="json"),
        )
        response.raise_for_status()

        trace_response = response.json()

        # Poll for evaluation result
        evaluation_result = await self.get_evaluation(
            trace_id=trace_response["trace_id"],
            wait=True,
        )

        return evaluation_result

    def evaluate_trace_sync(self, *args, **kwargs) -> EvaluationResult:
        """Synchronous version of evaluate_trace."""
        return asyncio.run(self.evaluate_trace(*args, **kwargs))

    async def evaluate_rl_episode(
        self,
        episode: List[Dict[str, Any]],
        policy_name: str,
        domain: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        Evaluate an RL episode.

        Args:
            episode: List of dicts with 'state', 'action', 'reward', 'next_state', 'done'
            policy_name: Name of the policy being evaluated
            domain: Domain (healthcare, trading, robotics)
            metadata: Additional metadata

        Returns:
            EvaluationResult
        """
        # Compute cumulative reward
        cumulative_reward = sum(step["reward"] for step in episode)

        return await self.evaluate_trace(
            input=f"RL episode with policy {policy_name}",
            output=f"Episode completed with cumulative reward: {cumulative_reward}",
            model=policy_name,
            domain=domain,
            application_type="rl_agent",
            rl_episode=episode,
            cumulative_reward=cumulative_reward,
            episode_length=len(episode),
            metadata=metadata,
        )

    async def get_evaluation(
        self, trace_id: str, wait: bool = False, timeout: float = 60.0
    ) -> EvaluationResult:
        """
        Get evaluation result for a trace.

        Args:
            trace_id: Trace ID
            wait: If True, poll until evaluation is complete
            timeout: Max time to wait (if wait=True)

        Returns:
            EvaluationResult
        """
        import time

        start_time = time.time()

        while True:
            response = await self._client.get(f"/api/v1/evaluations/{trace_id}")
            response.raise_for_status()

            data = response.json()

            # Check if evaluation is complete
            if data.get("completed_at") or not wait:
                return EvaluationResult(**data)

            # Check timeout
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Evaluation timed out after {timeout}s")

            # Wait before polling again
            await asyncio.sleep(1.0)

    async def batch_evaluate(
        self, traces: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple traces in batch.

        Args:
            traces: List of trace dicts (with input, output, model, etc.)

        Returns:
            List of EvaluationResults
        """
        tasks = [self.evaluate_trace(**trace) for trace in traces]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        return [r for r in results if isinstance(r, EvaluationResult)]

    async def create_domain(
        self, domain_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new domain from a specification.

        Args:
            domain_spec: Domain specification (YAML as dict)

        Returns:
            Created domain info
        """
        response = await self._client.post(
            "/api/v1/domains",
            json=domain_spec,
        )
        response.raise_for_status()

        return response.json()

    async def list_domains(self) -> List[str]:
        """List available domains."""
        response = await self._client.get("/api/v1/domains")
        response.raise_for_status()

        return response.json()["domains"]

    async def get_analytics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get analytics for your traces.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            domain: Filter by domain

        Returns:
            Analytics data
        """
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if domain:
            params["domain"] = domain

        response = await self._client.get("/api/v1/analytics", params=params)
        response.raise_for_status()

        return response.json()

    async def close(self) -> None:
        """Close HTTP clients."""
        await self._client.aclose()
        self._sync_client.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self._sync_client.close()

    async def __aenter__(self):
        """Async context manager support."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager cleanup."""
        await self.close()


# Integration helpers

class LangChainCallback:
    """
    LangChain callback for automatic Metronis evaluation.

    Usage:
        ```python
        from langchain.chains import LLMChain
        from metronis.sdk import LangChainCallback, MetronisClient

        client = MetronisClient(api_key="...")
        callback = LangChainCallback(client, domain="healthcare")

        chain = LLMChain(llm=llm, callbacks=[callback])
        result = chain.run("What medication for diabetes?")
        ```
    """

    def __init__(
        self,
        client: MetronisClient,
        domain: Optional[str] = None,
        application_type: Optional[str] = None,
    ):
        """Initialize the callback."""
        self.client = client
        self.domain = domain
        self.application_type = application_type

    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes."""
        input_text = kwargs.get("prompts", [""])[0]
        output_text = response.generations[0][0].text

        # Submit to Metronis
        asyncio.create_task(
            self.client.evaluate_trace(
                input=input_text,
                output=output_text,
                model=kwargs.get("invocation_params", {}).get("model_name", "unknown"),
                domain=self.domain,
                application_type=self.application_type,
            )
        )


class LlamaIndexCallback:
    """
    LlamaIndex callback for automatic Metronis evaluation.

    Usage:
        ```python
        from llama_index import ServiceContext, VectorStoreIndex
        from metronis.sdk import LlamaIndexCallback, MetronisClient

        client = MetronisClient(api_key="...")
        callback = LlamaIndexCallback(client, domain="healthcare")

        service_context = ServiceContext.from_defaults(callback_manager=callback)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        ```
    """

    def __init__(
        self,
        client: MetronisClient,
        domain: Optional[str] = None,
        application_type: str = "rag",
    ):
        """Initialize the callback."""
        self.client = client
        self.domain = domain
        self.application_type = application_type

    def on_event_end(self, event_type, payload, **kwargs):
        """Called when an event ends."""
        if event_type == "query":
            query = payload.get("query")
            response = payload.get("response")

            if query and response:
                asyncio.create_task(
                    self.client.evaluate_trace(
                        input=str(query),
                        output=str(response),
                        model=kwargs.get("model_name", "unknown"),
                        domain=self.domain,
                        application_type=self.application_type,
                    )
                )


# Convenience function
async def evaluate(
    input: str,
    output: str,
    model: str,
    api_key: str,
    domain: Optional[str] = None,
    **kwargs,
) -> EvaluationResult:
    """
    Quick evaluation function.

    Args:
        input: AI input
        output: AI output
        model: Model name
        api_key: Metronis API key
        domain: Domain name
        **kwargs: Additional parameters

    Returns:
        EvaluationResult
    """
    async with MetronisClient(api_key=api_key) as client:
        return await client.evaluate_trace(
            input=input,
            output=output,
            model=model,
            domain=domain,
            **kwargs,
        )
