---
name: software-architecture
description: Use when designing a flexible/scalable code structure for a pricing library (modules, interfaces, extensibility, packaging).
---
Workflow:
1) Propose a folder structure under src/pricing with clear boundaries:
   - instruments/, models/, methods/, marketdata/, api.py
2) Define interfaces (Protocols/ABCs) for Model, Pricer, MarketData.
3) Decide where validation, logging, and configuration live.
4) Ensure notebook uses public API only (no hidden notebook logic).
5) Identify future extension points (new payoff, new model, new method).
