---
name: senior-implementation
description: Use when writing production-quality Python implementation (clean API, error handling, readability, performance basics).
---
Workflow:
1) Implement smallest coherent slice end-to-end (API -> core -> tests).
2) Use type hints, dataclasses, and clear exceptions.
3) Avoid premature optimization; but keep vectorization and numerical stability in mind.
4) Keep functions small; add docstrings with units/assumptions.
5) Update README and ensure `pytest -q` passes.
