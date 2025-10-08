# Integrating Aegis with LlamaIndex

This guide shows you how to integrate Metronis Aegis with your LlamaIndex RAG applications.

## Installation

```bash
pip install metronis-aegis llama-index
```

## Basic Setup

```python
from aegis import AegisClient, Domain, RAGEvaluator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Initialize Aegis
aegis = AegisClient(
    api_key="your_api_key",
    domain=Domain.LEGAL
)

# Create LlamaIndex RAG pipeline
documents = SimpleDirectoryReader("./legal_documents").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Wrap with Aegis RAG evaluation
rag_eval = RAGEvaluator(client=aegis, domain=Domain.LEGAL)

@rag_eval.trace_rag()
async def query_legal_docs(question: str):
    response = query_engine.query(question)
    return response.response, response.source_nodes

# Run with automatic evaluation
result = await query_legal_docs("What are the contractual obligations in case of breach?")
```

## Advanced Features

### Custom Callback Integration

```python
from aegis.integrations.llamaindex import AegisCallbackHandler
from llama_index.core.callbacks import CallbackManager

callback = AegisCallbackHandler(client=aegis)
callback_manager = CallbackManager([callback])

index = VectorStoreIndex.from_documents(
    documents,
    callback_manager=callback_manager
)
```

### Multi-Step Agent Evaluation

```python
from llama_index.core.agent import ReActAgent
from aegis import WorkflowEvaluator

workflow_eval = WorkflowEvaluator(client=aegis, domain=Domain.FINANCIAL)

@workflow_eval.trace_workflow()
async def financial_analysis_agent(client_query: str):
    agent = ReActAgent.from_tools(tools, llm=llm)
    response = agent.chat(client_query)
    return response

# Aegis tracks tool calls, reasoning steps, and compliance
result = await financial_analysis_agent("Assess portfolio risk for conservative investor")
```

## Next Steps

- [Legal-specific LlamaIndex examples](../domains/legal.md)
- [Custom retrieval evaluators](../advanced/custom-evaluators.md)
- [Optimizing RAG performance](../advanced/performance.md)
