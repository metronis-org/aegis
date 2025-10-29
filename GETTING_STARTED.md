# Getting Started with Metronis Aegis

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd aegis

# Install Python dependencies
poetry install

# Start infrastructure services
docker-compose up -d

# Wait for services to be ready
docker-compose ps
```

### 2. Run the Demo

```bash
# Run the full system demo
poetry run python examples/demo_full_system.py
```

This demonstrates:
- Domain registration from YAML
- Auto-generation of evaluation modules
- LLM trace evaluation (clinical decision support)
- RL agent evaluation (trading)
- SDK usage

### 3. Start the API Server

```bash
# Run database migrations
poetry run alembic upgrade head

# Start the API server
poetry run uvicorn metronis.services.ingestion.app:app --reload

# API docs available at: http://localhost:8000/docs
```

### 4. Evaluate Your First Trace

```python
from metronis.sdk import MetronisClient

client = MetronisClient(
    api_key="demo-key",  # Get from admin
    base_url="http://localhost:8000"
)

# Evaluate a clinical decision
result = await client.evaluate_trace(
    input="Patient with hypertension, what medication?",
    output="Consider lisinopril 10mg once daily",
    model="gpt-4",
    domain="healthcare",
    application_type="clinical_support"
)

print(f"Evaluation passed: {result.overall_passed}")
print(f"Issues found: {len(result.all_issues)}")
for issue in result.all_issues:
    print(f"  - {issue.severity}: {issue.message}")
```

---

## Create Your Own Domain (15 minutes)

### Step 1: Define Domain Spec

Create `domains/my_domain/domain_spec.yaml`:

```yaml
domain_name: my_domain
display_name: My Custom Domain
description: Evaluation suite for my AI application
risk_level: high

entities:
  entity_type1: [field1, field2, field3]
  entity_type2: [field1, field2]

safety_constraints:
  - name: constraint_1
    description: Check that X does not exceed Y
    constraint_type: range_check
    parameters:
      max_value: 100
      field_name: score
    severity: high

  - name: constraint_2
    description: Validate entity exists in database
    constraint_type: existence_check
    parameters:
      knowledge_base: my_kb
    severity: critical

knowledge_bases:
  - name: my_kb
    type: rest_api
    api_url: https://api.example.com/
    cache_ttl: 3600

tier1_modules:
  - Constraint1Validator
  - Constraint2Validator

tier2_models:
  - name: RiskClassifier
    model_type: bert_classifier
    input_features: [feature1, feature2, feature3]
    output: risk_score
    training_data_source: expert_reviews

tier3_evals:
  - name: QualityEvaluator
    eval_type: quality_assessment
    criteria: [accuracy, completeness, safety]
    prompt_template: quality_eval_v1
```

### Step 2: Generate Modules

```bash
# Auto-generate evaluation modules from your spec
poetry run python -m metronis.cli generate-domain my_domain

# This creates:
# - domains/my_domain/tier1_modules/constraint_1_validator.py
# - domains/my_domain/tier1_modules/constraint_2_validator.py
# - domains/my_domain/tier2_models/risk_classifier.py
# - domains/my_domain/tier3_prompts/quality_evaluator.txt
```

### Step 3: Test Your Domain

```python
from metronis.sdk import MetronisClient

client = MetronisClient(api_key="demo-key")

result = await client.evaluate_trace(
    input="Test input for my domain",
    output="Test output from my AI",
    model="my-model",
    domain="my_domain",
    metadata={
        "custom_field1": "value1",
        "entities": [
            {"type": "entity_type1", "field1": "val1"}
        ]
    }
)

print(f"Passed: {result.overall_passed}")
```

---

## Evaluate an RL Agent (10 minutes)

### Step 1: Create RL Episode

```python
from metronis.sdk import MetronisClient

# Your RL agent's episode
episode = [
    {
        "state": {"position": 0, "velocity": 0},
        "action": {"acceleration": 1.0},
        "reward": 1.0,
        "next_state": {"position": 1, "velocity": 1},
        "done": False
    },
    {
        "state": {"position": 1, "velocity": 1},
        "action": {"acceleration": 0.5},
        "reward": 2.0,
        "next_state": {"position": 2, "velocity": 1.5},
        "done": False
    },
    {
        "state": {"position": 2, "velocity": 1.5},
        "action": {"acceleration": 0},
        "reward": 5.0,
        "next_state": {"position": 3, "velocity": 1.5},
        "done": True
    }
]
```

### Step 2: Evaluate Episode

```python
client = MetronisClient(api_key="demo-key")

result = await client.evaluate_rl_episode(
    episode=episode,
    policy_name="MyRLAgent-v1",
    domain="robotics",  # or healthcare, trading, etc.
    metadata={
        "environment": "CartPole-v1",
        "training_step": 10000
    }
)

print(f"Episode evaluated: {result.trace_id}")
print(f"Cumulative reward: {sum(step['reward'] for step in episode)}")
print(f"Safety issues: {len([i for i in result.all_issues if i.severity == 'CRITICAL'])}")

# Check RL-specific evaluations
for module_result in result.tier2_results:
    if "reward" in module_result.module_name.lower():
        print(f"Reward shaping check: {module_result.passed}")
    if "exploration" in module_result.module_name.lower():
        print(f"Exploration efficiency: {module_result.metadata}")
```

---

## Integrate with LangChain (5 minutes)

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from metronis.sdk import MetronisClient, LangChainCallback

# Initialize Metronis
client = MetronisClient(api_key="your-key")
metronis_callback = LangChainCallback(
    client=client,
    domain="healthcare",
    application_type="clinical_support"
)

# Create LangChain chain
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = PromptTemplate(
    input_variables=["patient_info"],
    template="Based on this patient information: {patient_info}\n\nWhat treatment do you recommend?"
)
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    callbacks=[metronis_callback]  # Automatic evaluation
)

# Run chain - automatically evaluated by Metronis
result = chain.run(patient_info="65yo male, hypertension, BP 150/95")

# Check evaluation results
evaluations = await client.get_analytics(domain="healthcare")
print(f"Total traces: {evaluations['total_traces']}")
print(f"Pass rate: {evaluations['pass_rate']:.1%}")
```

---

## Integrate with LlamaIndex (5 minutes)

```python
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from metronis.sdk import MetronisClient, LlamaIndexCallback

# Initialize Metronis
client = MetronisClient(api_key="your-key")
metronis_callback = LlamaIndexCallback(
    client=client,
    domain="legal",
    application_type="rag"
)

# Create LlamaIndex service context
llm = OpenAI(model="gpt-4", temperature=0)
service_context = ServiceContext.from_defaults(
    llm=llm,
    callback_manager=metronis_callback
)

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)

# Query - automatically evaluated
query_engine = index.as_query_engine()
response = query_engine.query("What are the precedents for this case?")

# Every query is automatically evaluated for:
# - Citation accuracy
# - Retrieval quality
# - Hallucination detection
# - Legal reasoning
```

---

## Advanced: Train Tier-2 ML Model (30 minutes)

### Step 1: Collect Training Data

```python
# Expert reviews provide training labels
from metronis.core.domain import DomainRegistry

registry = DomainRegistry(Path("domains"))
healthcare = registry.get_domain("healthcare")

# Get expert-reviewed traces
# (Tier 4 creates labeled training data)
training_data = fetch_expert_reviews(domain="healthcare", limit=1000)

# Format: [{"features": {...}, "label": 0/1}, ...]
```

### Step 2: Train Model

```python
from domains.healthcare.tier2_models.safety_risk_classifier import SafetyRiskClassifier
from pathlib import Path

# Initialize model
model = SafetyRiskClassifier(
    model_path=Path("models/healthcare/safety_classifier"),
    config={
        "fine_tune_epochs": 10,
        "batch_size": 32,
        "model_size": "base"
    }
)

# Train
model.train(
    training_data=training_data[:800],
    validation_data=training_data[800:]
)

# Model auto-saved to model_path
print("Model trained and saved!")
```

### Step 3: Deploy Model

```python
# Load trained model
model = SafetyRiskClassifier(
    model_path=Path("models/healthcare/safety_classifier")
)

# Use in evaluation
from metronis.core.orchestrator import ModuleRegistry

module_registry = ModuleRegistry()
module_registry.register_module(model, "healthcare", tier=2)

# Now automatically used in Tier 2 evaluations
```

---

## Monitor & Analytics (5 minutes)

### View Metrics

```python
from metronis.sdk import MetronisClient

client = MetronisClient(api_key="your-key")

# Get overall analytics
analytics = await client.get_analytics(
    start_date="2025-01-01",
    end_date="2025-01-31",
    domain="healthcare"
)

print(f"Total traces: {analytics['total_traces']}")
print(f"Pass rate: {analytics['pass_rate']:.1%}")
print(f"Average cost per trace: ${analytics['avg_cost_per_trace']:.3f}")
print(f"P95 latency: {analytics['p95_latency_ms']}ms")

# Error breakdown
for error_type, count in analytics['error_types'].items():
    print(f"  {error_type}: {count} occurrences")

# Tier coverage
print(f"\nTier coverage:")
print(f"  Tier 1 only: {analytics['tier1_only_pct']:.1%}")
print(f"  Escalated to Tier 2: {analytics['tier2_pct']:.1%}")
print(f"  Escalated to Tier 3: {analytics['tier3_pct']:.1%}")
print(f"  Expert review (Tier 4): {analytics['tier4_pct']:.1%}")
```

### Grafana Dashboards

```bash
# Access Grafana
open http://localhost:3000

# Default credentials: admin/admin

# Pre-built dashboards:
# 1. System Overview - Traces/sec, latency, error rates
# 2. Cost Analysis - Cost per trace, tier distribution
# 3. Domain Performance - Pass rates by domain
# 4. RL Metrics - Reward distributions, exploration efficiency
```

---

## Troubleshooting

### Common Issues

**1. "Domain not found" error**

```bash
# Check loaded domains
poetry run python -c "
from metronis.core.domain import DomainRegistry
from pathlib import Path
registry = DomainRegistry(Path('domains'))
print('Loaded domains:', registry.list_domains())
"
```

**2. Knowledge base API errors**

```python
# Test knowledge base connectivity
from metronis.services.knowledge_base_service import KnowledgeBaseService

kb_service = KnowledgeBaseService()
result = await kb_service.check_existence("rxnorm", "lisinopril")
print(f"RxNorm working: {result['exists']}")
```

**3. Redis connection failed**

```bash
# Check Redis is running
docker-compose ps redis

# Test connection
redis-cli ping
# Should return: PONG
```

---

## Next Steps

1. **Customize your domain** - Modify `domain_spec.yaml` for your use case
2. **Train Tier-2 models** - Collect expert reviews, train classifiers
3. **Set up production** - Deploy to Kubernetes, configure monitoring
4. **Integrate with your app** - Use SDK or callbacks
5. **Monitor performance** - Track metrics, optimize thresholds

**Questions?** See the [System Architecture](SYSTEM_ARCHITECTURE.md) or join our [Discord](https://discord.gg/metronis)

**Ready to deploy?** See [Deployment Guide](docs/deployment.md)
