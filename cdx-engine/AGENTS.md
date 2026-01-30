# AGENTS.md — Quant Pricing Project Rules

## Goals
- Deliver: (1) pricing library in src/, (2) tests in tests/ (pytest), (3) notebook demo in notebooks/ with plots, (4) short README.

## Engineering standards
- Python only. Prefer pure functions + dataclasses; keep side effects in CLI/notebook.
- Every pricing function must have:
  - clear docstring with units/assumptions
  - input validation
  - at least 2 pytest tests: sanity + edge case
- Before finishing: run pytest -q and ensure it passes.

## Quant standards
- Always state model assumptions (measure, dynamics, rates/dividends, day count, etc.).
- Provide: price + Greeks (where applicable), and at least one calibration or parameter sweep example.
- Numerical methods must include stability notes (grid/time step, MC variance reduction, etc.).

## Notebook deliverable
- Provide a single notebook: notebooks/demo.ipynb
- Must show:
  - model parameters
  - price/Greeks table
  - at least 2 plots (e.g., price surface, sensitivity curve)
- Plots: matplotlib preferred; keep it reproducible.

## Architecture
- Keep core library in src/pricing/ and avoid notebook-only logic.
- Add a clean API facade: src/pricing/api.py

## Working style
- Small, reviewable diffs. Use /diff frequently.
- After implementation, run /review to catch missing tests or issues.
# Repository Guidelines


## Project Structure & Module Organization
- `src/` holds the core pricing engine modules (curves, copula, calibration, risk, hedging).
- `tests/` contains unit tests (pytest-style) that exercise key math and calibration logic.
- `configs/` stores runtime configuration (e.g., `configs/params.yaml`).
- `data/` contains input CSVs used by scripts and notebooks.
- `scripts/` provides small batch entrypoints (e.g., `scripts/run_daily.py`).
- `notebooks/` includes analysis and workflow notebooks.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs Python dependencies.
- `python -m pytest` runs the test suite in `tests/`.
- `python scripts/run_daily.py` runs the daily data QC/ingest example using `configs/params.yaml`.
- `python scripts/run_loocv.py` runs the leave-one-out CV example.

## Coding Style & Naming Conventions
- Language: Python, with standard 4-space indentation.
- Use snake_case for functions and modules; class names use CapWords (e.g., `Curve`).
- Prefer explicit typing where it improves clarity (the codebase already uses type hints in places).
- Keep modules focused (one pricing or utility concern per file in `src/`).

## Testing Guidelines
- Framework: pytest conventions (`test_*.py`, `test_*` functions).
- Keep tests deterministic and lightweight; avoid randomness or external services.
- Place shared fixtures or path setup in `tests/conftest.py`.

## Commit & Pull Request Guidelines
- Commit history uses short, sometimes non‑English messages and does not follow a strict convention.
- For new work, use clear, imperative subject lines (e.g., “Add base correlation surface interpolation”).
- PRs should include: a brief summary, test results (commands + outcome), and any data/config changes.

## Configuration & Data Tips
- Scripts assume local CSVs in `data/` and parameters in `configs/params.yaml`.
- If you change data formats or schema, update the relevant readers in `src/io_data.py` and add tests.
