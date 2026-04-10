"""
backtest — Walk-forward purged cross-validation backtester for SentiStack V2.

Task 2 (scaffold, part 1/2) delivers:
  • cost_model.py  — bit-for-bit mirror of logbook.py + strategy.py costs
  • data_loader.py — Zerodha historical minute-bar fetch + parquet cache
  • engine.py      — purged walk-forward splitter + event-driven replay
                     with a dummy always-flat strategy for end-to-end test

Task 3 will add strategy_adapter.py that wraps the live StrategyEngine so
the same risk manager and portfolio budgets can be validated against history.
"""

__all__ = ["cost_model", "data_loader", "engine"]
