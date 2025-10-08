<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" alt="Aegis logo" height="150" srcset="docs/img/logo_white.png">
    <source media="(prefers-color-scheme: light)" alt="Aegis logo" height="150" srcset="docs/img/logo_black.png">
    <img alt="Aegis logo" height="150" src="docs/img/logo_black.png">
  </picture>

  [![Python](https://img.shields.io/badge/Python-333333?logo=python&logoColor=white&labelColor=333333)](#)
  [![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=chainlink&logoColor=white)](#)
  [![LlamaIndex](https://img.shields.io/badge/LlamaIndex-8B5CF6?logo=databricks&logoColor=white)](#)
  [![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white)](https://discord.gg/metronis-aegis)
  <br>
  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![HIPAA](https://img.shields.io/badge/HIPAA-Compliant-4A90E2)](#)
  [![SOC2](https://img.shields.io/badge/SOC2-Type%20II-00D084)](#)
</div>

> 🚀 **Production Beta Now Open** - Join leading healthcare, legal, and financial institutions using Aegis to deploy AI agents safely in regulated environments.  
>> 🎯 **Enterprise Program**: Book a demo at [metronis.ai/demo](https://metronis.ai/demo)

**aegis** is the first domain-specific evaluation platform for AI agents in regulated industries. We don't build agents—we build the evaluation infrastructure that ensures every agent operates safely and compliantly in healthcare, legal, and financial sectors.

With Aegis, you can:
- automatically evaluate AI agents across **healthcare, legal, and financial** domains with domain-specific expertise
- detect **compliance violations and safety issues** that generic tools miss using ML-based clustering and expert-labeled datasets
- get **actionable fix suggestions** with A/B testing in simulation before production
- ensure **regulatory compliance** (HIPAA, SOC2, GDPR) with built-in audit trails and human-in-the-loop review

## Why Aegis?

| Generic Tools (Arize, Braintrust) | **Metronis Aegis** |
|---|---|
| ❌ Horizontal observability | ✅ **Domain expertise** in healthcare, legal, finance |
| ❌ Generic eval templates | ✅ **Automated eval generation** from production failures |
| ❌ Manual eval creation | ✅ **Active learning flywheel** - improves with usage |
| ❌ No compliance focus | ✅ **Regulatory compliance** built into every layer |
| ❌ "Figure it out yourself" | ✅ **Root cause + fixes** with A/B tested suggestions |

**In regulated industries, "move fast and break things" isn't an option. That difference is worth 10x the price.**

<br/>

# Quick Start

- [Get started with Healthcare Agents](https://docs.metronis.ai/aegis/quickstart-healthcare)
- [Get started with Legal Agents](https://docs.metronis.ai/aegis/quickstart-legal)
- [Get started with Financial Agents](https://docs.metronis.ai/aegis/quickstart-financial)
- [Get started with the Python SDK](https://docs.metronis.ai/aegis/quickstart-sdk)

<br/>

# Usage ([Docs](https://docs.metronis.ai/aegis))

```bash
pip install metronis-aegis[all]
```
```python
from aegis import AegisClient, Domain

# Initialize with your domain
client = AegisClient(
    api_key="your_api_key",
    domain=Domain.HEALTHCARE  # or LEGAL, FINANCIAL
)

# Wrap your agent function
@client.trace()
async def diagnose_patient(symptoms: str, history: dict):
    response = await your_llm_call(symptoms, history)
    return response

# Run with automatic evaluation
result = await diagnose_patient(
    symptoms="persistent headache, dizziness",
    history={...}
)

# Get evaluation results
eval = client.get_evaluation(result.trace_id)
print(f"Safety Score: {eval.safety_score}")
print(f"Compliance: {eval.compliance_status}")
print(f"Suggestions: {eval.suggestions}")
```

### Output format:
```json
{
  "trace_id": "trace_abc123",
  "evaluation": {
    "safety_score": 0.94,
    "compliance_status": "PASSED",
    "domain_metrics": {
      "medical_terminology_accuracy": 0.97,
      "hipaa_compliance": true,
      "clinical_guideline_adherence": 0.92
    },
    "detected_issues": [
      {
        "severity": "medium",
        "type": "missing_context",
        "description": "Patient allergy history not considered",
        "suggestion": "Add allergy check before medication recommendation"
      }
    ],
    "fix_suggestions": [
      {
        "type": "prompt_refinement",
        "priority": "high",
        "change": "Add: 'Always verify patient allergies before recommendations'",
        "estimated_improvement": "+8% safety score",
        "ab_test_result": "95% confidence, +12% reduction in errors"
      }
    ]
  },
  "usage": {
    "total_tokens": 1247,
    "evaluation_cost": 0.03,
    "duration_ms": 2341
  }
}
```

# Features

## 🔄 Automated QA Pipeline

1. **Trace Collection** - Zero-friction SDK captures every LLM call, tool invocation, token usage, cost, and latency
2. **Error Detection** - ML clustering + domain-specific signals (medical terminology, legal citations, regulatory patterns)
3. **Eval Generation** - Auto-creates LLM-as-judge evaluators calibrated on expert-labeled data, improved via RL
4. **Fix Suggestions** - Root cause analysis → A/B tested recommendations → production-ready improvements
5. **Human Review** - Expert oversight for high-stakes decisions with audit trails

## 🎓 Reinforcement Learning Layer

Our evaluation accuracy continuously improves through RL environments simulating:
- **Healthcare**: Patient consultations, diagnosis workflows, treatment planning
- **Legal**: Case analysis, contract review, compliance queries
- **Financial**: Risk assessment, fraud detection, advisory decisions

**We train our evaluation engine to be the most accurate domain expert—not your agents.**

## 🏆 Technical Moat

- **Proprietary Algorithms**: Domain-aware clustering + multi-objective calibration (safety, explainability, compliance)
- **Expert Dataset**: Exclusive partnerships providing domain-specific failure patterns
- **Active Learning**: Each trace improves accuracy → more usage = better evals across all customers

<br/>

# Use Cases

<table>
<tr>
<td width="33%">

### 🏥 Healthcare
**Diagnostic Assistant**
- Medical terminology validation
- Treatment recommendation checks
- HIPAA compliance
- Clinical guideline adherence

*Example*: Hospital AI triage ensures symptoms aren't missed and critical conditions are properly escalated.

</td>
<td width="33%">

### ⚖️ Legal
**Legal Research Tool**
- Case law citation validation
- Legal reasoning soundness
- Regulatory compliance
- Multi-step analysis tracking

*Example*: Law firm contract agent verifies obligations are identified and risks are flagged.

</td>
<td width="33%">

### 💰 Financial
**Advisory Agent**
- Fraud pattern detection
- SEC/FINRA compliance
- Risk assessment validation
- Investment recommendation review

*Example*: Fintech robo-advisor monitors fiduciary compliance and detects risky recommendations.

</td>
</tr>
</table>

<br/>

# Domains

## Supported Architectures

**We evaluate complex multi-turn interactions:**
- ✅ Agentic workflows with multiple tool calls
- ✅ RAG pipelines (retrieval + generation)
- ✅ Multi-step reasoning chains
- ✅ Context-dependent decision making

## Current Integrations

Starting with the **80% of our ICP** building on:
- 🐍 **Python SDK** (primary)
- 🦜 **LangChain**
- 🦙 **LlamaIndex**

*Additional frameworks added based on customer demand.*

<br/>

# Modules

| Module | Description | Installation |
|--------|-------------|---------------|
| [**Aegis Core**](./libs/python/core/README.md) | Core evaluation engine and trace collection | `pip install metronis-aegis` |
| [**Aegis Healthcare**](./libs/python/healthcare/README.md) | Healthcare-specific evaluators and compliance | `pip install "metronis-aegis[healthcare]"` |
| [**Aegis Legal**](./libs/python/legal/README.md) | Legal domain evaluators and citation validation | `pip install "metronis-aegis[legal]"` |
| [**Aegis Financial**](./libs/python/financial/README.md) | Financial services compliance and risk detection | `pip install "metronis-aegis[financial]"` |
| [**Aegis SDK**](./libs/python/sdk/README.md) | Python SDK for agent integration | `pip install aegis-sdk` |
| [**Aegis Dashboard**](./libs/dashboard/README.md) | Web dashboard for evaluation monitoring | `npm install @metronis/aegis-dashboard` |

<br/>

# Resources

- [How to integrate Aegis with LangChain agents](./docs/integrations/langchain.md)
- [How to use Aegis with LlamaIndex RAG pipelines](./docs/integrations/llamaindex.md)
- [Healthcare Compliance Guide: HIPAA and Clinical Guidelines](./docs/domains/healthcare-compliance.md)
- [Legal Evaluation: Citation Validation and Case Law Analysis](./docs/domains/legal-evaluation.md)
- [Financial Services: Fraud Detection and Regulatory Compliance](./docs/domains/financial-compliance.md)
- [Building Custom Evaluators for Your Domain](./docs/advanced/custom-evaluators.md)

<br/>

# Security & Compliance

<div align="center">

| **SOC 2 Type II** | **HIPAA** | **GDPR** | **ISO 27001** |
|:-----------------:|:---------:|:--------:|:-------------:|
| ✅ Certified | ✅ Compliant | ✅ Compliant | ✅ Certified |

</div>

- End-to-end encryption for all traces
- Audit trails for regulatory review
- Data residency options (US, EU, APAC)
- SSO/SAML support
- Zero data retention options available

<br/>

# Community

Join our [Discord community](https://discord.gg/metronis-aegis) to discuss ideas, get assistance, share your demos, and connect with other teams building AI agents in regulated industries!

<br/>

# Contributing

We welcome contributions from domain experts and engineers! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Areas We Need Help
- Domain-specific evaluation criteria
- Framework integrations (Haystack, Semantic Kernel, AutoGen)
- Performance optimizations
- Documentation
- Expert-labeled datasets for new domains

<br/>

# Roadmap

<table>
<tr>
<td width="33%">

### Q4 2025
- [ ] Public beta launch
- [ ] LangGraph integration
- [ ] Real-time dashboard v2
- [ ] Extended healthcare modules

</td>
<td width="33%">

### Q1 2026
- [ ] Government sector support
- [ ] Multi-language (ES, DE, FR)
- [ ] Advanced A/B testing
- [ ] Custom RL builder

</td>
<td width="33%">

### Q2 2026
- [ ] On-premise deployment
- [ ] Advanced analytics
- [ ] Third-party audit integration
- [ ] Expanded tool ecosystem

</td>
</tr>
</table>

<br/>

# License

Metronis Aegis is open-sourced under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

Some optional extras depend on third-party packages licensed under different terms. When you install optional extras, your use is governed by their respective licenses.

<br/>

# Trademarks

LangChain is a trademark of LangChain, Inc.  
LlamaIndex is a trademark of LlamaIndex.  
Anthropic and Claude are trademarks of Anthropic, PBC.  
OpenAI is a trademark of OpenAI, Inc.

This project is not affiliated with, endorsed by, or sponsored by any of the above companies.

<br/>

## Stargazers

Thank you to all our supporters!

[![Stargazers over time](https://starchart.cc/metronis-org/aegis.svg?variant=adaptive)](https://starchart.cc/metronis-org/aegis)

---

<div align="center">

**Built with ❤️ by the Metronis team**

Making AI agents safe for regulated industries

[Website](https://metronis.ai) • [Blog](https://metronis.ai/blog) • [Twitter](https://twitter.com/metronisai) • [LinkedIn](https://linkedin.com/company/metronis) • [Discord](https://discord.gg/metronis-aegis)

*Trusted by leading healthcare, legal, and financial institutions worldwide*

</div>
