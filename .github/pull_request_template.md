## Description
Briefly describe the purpose of this PR and what issues it closes.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature that alters API contracts)
- [ ] Model weights update or retraining logic enhancement
- [ ] Documentation update

## Pre-Merge Checklist
- [ ] My code adheres to PEP 8 standards and passes `black --check src tests`
- [ ] Static type checks pass via `mypy src configs`
- [ ] All integration tests pass via `pytest tests/`
- [ ] New endpoints include role-based authentication and Prometheus instrumentation
- [ ] Sensitive secrets and `.env` files are excluded from commits
