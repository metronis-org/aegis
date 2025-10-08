# Contributing to Metronis Aegis

First off, thank you for considering contributing to Aegis! It's people like you that make Aegis such a great tool for the AI community in regulated industries.

## 🌟 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** to demonstrate the steps
- **Describe the behavior you observed** and what you expected to see
- **Include screenshots or code snippets** if relevant
- **Specify your environment**: OS, Python version, Aegis version, domain (healthcare/legal/financial)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful** to most Aegis users
- **List any alternative solutions** you've considered

### Pull Requests

1. **Fork the repo** and create your branch from `main`
2. **Install dependencies**: `pip install -e ".[dev]"`
3. **Make your changes** and add tests if applicable
4. **Ensure tests pass**: `pytest tests/`
5. **Follow the code style**: `black . && ruff check .`
6. **Update documentation** if you changed APIs
7. **Write a clear commit message**

## 💻 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/aegis.git
cd aegis

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black .
ruff check .
```

## 🧪 Testing

We use `pytest` for testing. Please ensure all tests pass before submitting a PR:

```bash
# Run all tests
pytest

# Run tests for a specific domain
pytest tests/healthcare/
pytest tests/legal/
pytest tests/financial/

# Run with coverage
pytest --cov=aegis tests/
```

## 📝 Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for all function signatures
- **Docstrings** in Google style

Example:
```python
from typing import Dict, List, Optional

def evaluate_trace(
    trace_id: str,
    domain: str,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate a trace using domain-specific metrics.
    
    Args:
        trace_id: Unique identifier for the trace
        domain: Domain type (healthcare, legal, financial)
        metrics: Optional list of specific metrics to evaluate
        
    Returns:
        Dictionary mapping metric names to scores
        
    Raises:
        ValueError: If domain is not supported
    """
    pass
```

## 🏗️ Project Structure

```
aegis/
├── libs/
│   └── python/
│       ├── core/           # Core evaluation engine
│       ├── healthcare/     # Healthcare-specific evaluators
│       ├── legal/          # Legal-specific evaluators
│       ├── financial/      # Financial-specific evaluators
│       └── sdk/            # Python SDK
├── docs/                   # Documentation
├── tests/                  # Test suite
└── examples/               # Example integrations
```

## 🔍 Areas We Need Help

### Domain Expertise
- **Healthcare**: Clinical guidelines, medical terminology validation
- **Legal**: Case law citations, regulatory compliance
- **Financial**: Fraud detection patterns, regulatory requirements

### Technical Contributions
- **Framework integrations**: Haystack, Semantic Kernel, AutoGen
- **Performance optimizations**: Caching, async improvements
- **Documentation**: Tutorials, guides, API docs
- **Testing**: Increase coverage, add edge cases

### Dataset Contributions
We're actively seeking **expert-labeled datasets** for:
- Medical diagnosis conversations
- Legal case analysis
- Financial advisory interactions

*Note: All dataset contributions require proper licensing and privacy compliance.*

## 🤝 Community Guidelines

- **Be respectful** and constructive in discussions
- **Help others** when you can
- **Ask questions** if something is unclear
- **Share your use cases** to help improve Aegis
- **Follow the** [Code of Conduct](CODE_OF_CONDUCT.md)

## 📋 Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add support for LangGraph integration
fix: resolve HIPAA compliance check error
docs: update healthcare evaluation guide
test: add tests for legal citation validation
refactor: improve error detection clustering
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

## 🎯 Domain-Specific Contributions

### Healthcare
- Must comply with HIPAA guidelines
- Clinical accuracy is paramount
- Cite medical sources when applicable

### Legal
- Follow legal citation standards (Bluebook, etc.)
- Ensure regulatory compliance
- Verify case law accuracy

### Financial
- Comply with SEC, FINRA regulations
- Implement fraud detection best practices
- Follow financial industry standards

## 📞 Questions?

- **Discord**: [Join our server](https://discord.gg/metronis-aegis)
- **Email**: developers@metronis.ai
- **Discussions**: [GitHub Discussions](https://github.com/metronis-org/aegis/discussions)

---

Thank you for contributing to making AI safer in regulated industries! 🛡️
