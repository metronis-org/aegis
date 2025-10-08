# Integrating Aegis with LangChain

This guide shows you how to integrate Metronis Aegis with your LangChain agents.

## Installation

```bash
pip install metronis-aegis langchain
```

## Basic Setup

```python
from aegis import AegisClient, Domain
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Initialize Aegis
aegis = AegisClient(
    api_key="your_api_key",
    domain=Domain.HEALTHCARE
)

# Create LangChain agent
llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([...])
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Wrap with Aegis tracing
@aegis.trace()
async def run_agent(query: str):
    return await agent_executor.ainvoke({"input": query})

# Run with automatic evaluation
result = await run_agent("What is the recommended treatment for hypertension?")
```

## Advanced Features

### Custom Callbacks

```python
from aegis.integrations.langchain import AegisCallbackHandler

callback = AegisCallbackHandler(client=aegis)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[callback]
)
```

### RAG Evaluation

```python
from aegis import RAGEvaluator
from langchain.chains import RetrievalQA

rag_eval = RAGEvaluator(client=aegis, domain=Domain.LEGAL)

@rag_eval.trace_rag()
async def legal_qa(question: str):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )
    return await qa_chain.ainvoke(question)
```

## Next Steps

- [Healthcare-specific LangChain examples](../domains/healthcare.md)
- [Custom evaluators for LangChain](../advanced/custom-evaluators.md)
- [Performance optimization tips](../advanced/performance.md)
