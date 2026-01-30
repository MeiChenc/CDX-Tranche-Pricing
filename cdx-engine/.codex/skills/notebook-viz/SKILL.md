---
name: notebook-viz
description: Use when producing a final Jupyter notebook demo with plots and result tables for a pricing model.
---
Workflow:
1) Create `notebooks/demo.ipynb` that imports only from `src/pricing/api.py`.
2) Show: parameters, price+Greeks table (pandas), and at least 2 plots (matplotlib).
3) Include one sensitivity sweep (e.g., sigma or T) + visualization.
4) Keep it reproducible: fixed seeds for MC, clear cell order, minimal hidden state.
5) End with a short "Summary" cell listing assumptions + limitations.
