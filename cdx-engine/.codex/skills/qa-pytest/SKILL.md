---
name: qa-pytest
description: Use when creating or improving automated tests for pricing models using pytest (unit tests, properties, edge cases).
---
Workflow:
1) Add sanity tests (known closed-form, parity, monotonicity, bounds).
2) Add edge cases (T->0, sigma->0, deep ITM/OTM, negative rates if supported).
3) Add numerical regression tests with tolerances (avoid brittle exact equality).
4) Ensure tests run fast; mark slow tests and keep defaults quick.
5) Suggest CI-ready commands (pytest -q).
