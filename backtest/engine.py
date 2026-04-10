"""
backtest/engine.py
==================

Event-driven backtest engine with purged walk-forward cross-validation.

Why purged walk-forward?
------------------------
Standard k-fold CV leaks future information into the training set
through overlapping labels (a bar's label depends on bars N minutes
ahead, so the test fold's "future" is already visible in the training
fold). López de Prado's purged & embargoed walk-forward CV fixes this:

    1. Split the timeline into k contiguous folds (not random).
    2. For each fold, use all bars BEFORE it as training.
    3. "Purge" training bars whose labels extend into the test window.
    4. "Embargo" a small window AFTER the test set to prevent serial
       correlation leaking the other direction.

Task 2 (this file) ships the scaffold:
    • `PurgedWalkForward` splitter
    • Event-driven `BacktestEngine` that replays bars in order
    • `Strategy` protocol for swappable strategies
    • `DummyFlatStrategy` — always flat, proves the pipeline runs
      end-to-end without needing the live StrategyEngine yet
    • CLI entrypoint: `python -m backtest.engine --symbol RELIANCE ...`

Task 3 will add `strategy_adapter.py` that wraps the live
`StrategyEngine` and produces real Kelly-sized trades so the same
risk manager used in production can be validated against history.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

import pandas as pd

from .cost_model import leg_cost
from .data_loader import generate_synthetic_bars, load_bars

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Purged walk-forward splitter
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, expressed as integer row indices."""
    k:           int
    train_start: int
    train_end:   int   # exclusive
    test_start:  int
    test_end:    int   # exclusive
    embargo_end: int   # exclusive — training for the NEXT fold must skip up to here


class PurgedWalkForward:
    """
    Purged & embargoed walk-forward splitter.

    Parameters
    ----------
    n_splits : int
        Number of contiguous test folds.
    embargo_pct : float
        Fraction of total samples to blank out after each test window.
        0.02 (2%) is a reasonable default for minute-bar intraday data.
    purge_bars : int
        Number of bars to drop from the END of each training window to
        prevent leakage from overlapping labels. Defaults to 0 for the
        scaffold — Task 3 will set this from the strategy's label horizon.
    """

    def __init__(
        self,
        n_splits:    int   = 5,
        embargo_pct: float = 0.02,
        purge_bars:  int   = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 <= embargo_pct < 0.5:
            raise ValueError("embargo_pct must be in [0, 0.5)")
        if purge_bars < 0:
            raise ValueError("purge_bars must be >= 0")
        self.n_splits    = n_splits
        self.embargo_pct = embargo_pct
        self.purge_bars  = purge_bars

    def split(self, n_samples: int) -> List[Fold]:
        if n_samples < self.n_splits * 2:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits"
            )

        embargo = int(n_samples * self.embargo_pct)
        fold_size = n_samples // self.n_splits
        folds: List[Fold] = []

        # First fold uses the first fold_size rows purely as a "warm-up" train,
        # and the second fold_size as test. Subsequent folds walk forward.
        for k in range(1, self.n_splits):
            test_start  = k * fold_size
            test_end    = test_start + fold_size if k < self.n_splits - 1 else n_samples
            train_start = 0
            train_end   = max(0, test_start - self.purge_bars)
            embargo_end = min(n_samples, test_end + embargo)
            folds.append(Fold(
                k=k,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_end=embargo_end,
            ))
        return folds


# ---------------------------------------------------------------------------
# Strategy protocol + dummy implementation
# ---------------------------------------------------------------------------
@dataclass
class Signal:
    """A single strategy output at one bar."""
    timestamp: pd.Timestamp
    symbol:    str
    action:    str          # "BUY" | "SELL" | "HOLD"
    qty:       int  = 0
    reason:    str  = ""


class Strategy(Protocol):
    """Minimal strategy contract. Task 3's adapter will satisfy this too."""

    def on_bar(self, ts: pd.Timestamp, symbol: str, bar: pd.Series) -> Optional[Signal]:
        ...

    def reset(self) -> None:
        ...


class DummyFlatStrategy:
    """Always flat. Used to prove the event loop + reporting work."""

    def on_bar(self, ts: pd.Timestamp, symbol: str, bar: pd.Series) -> Optional[Signal]:
        return None

    def reset(self) -> None:
        return None


class ScaffoldMomentumStrategy:
    """
    Tiny momentum-style strategy used ONLY to verify that non-flat
    trades flow through the engine, cost model, and report correctly.
    NOT meant to make money — Task 3 replaces this with the real
    StrategyEngine adapter.

    Rule: on every Nth bar, if the close > close N bars ago, go long
    1 share for the next bar; otherwise flat.
    """

    def __init__(self, lookback: int = 20, stride: int = 30) -> None:
        self.lookback  = lookback
        self.stride    = stride
        self._history: List[float] = []
        self._bar_i    = 0
        self._position = 0

    def reset(self) -> None:
        self._history.clear()
        self._bar_i    = 0
        self._position = 0

    def on_bar(self, ts: pd.Timestamp, symbol: str, bar: pd.Series) -> Optional[Signal]:
        self._history.append(float(bar["close"]))
        self._bar_i += 1
        if len(self._history) <= self.lookback:
            return None
        if self._bar_i % self.stride != 0:
            return None

        close_now  = self._history[-1]
        close_prev = self._history[-1 - self.lookback]

        if self._position == 0 and close_now > close_prev:
            self._position = 1
            return Signal(ts, symbol, "BUY", qty=1, reason="momentum_up")
        if self._position == 1 and close_now < close_prev:
            self._position = 0
            return Signal(ts, symbol, "SELL", qty=1, reason="momentum_down")
        return None


# ---------------------------------------------------------------------------
# Event-driven engine
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    entry_ts:   pd.Timestamp
    exit_ts:    Optional[pd.Timestamp]
    symbol:     str
    entry_px:   float
    exit_px:    Optional[float]
    qty:        int
    cost:       float  = 0.0      # round-trip INR cost
    gross_pnl:  float  = 0.0
    net_pnl:    float  = 0.0
    reason:     str    = ""


@dataclass
class BacktestResult:
    symbol:       str
    fold:         int
    n_bars:       int
    n_trades:     int
    gross_pnl:    float
    net_pnl:      float
    total_costs:  float
    win_rate:     float
    profit_factor: float          # sum(wins) / |sum(losses)|, ∞ if no losses
    sharpe:       float
    max_dd:       float
    trades:       List[Trade] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("trades", None)
        return d


class BacktestEngine:
    """
    Minimal event-driven replay loop.

    For each bar:
      1. Mark-to-market any open position.
      2. Call strategy.on_bar(ts, symbol, bar).
      3. If signal: flatten existing / open new at next bar's open
         (to avoid same-bar look-ahead bias).
    """

    def __init__(self, strategy: Strategy, product: str = "MIS") -> None:
        self.strategy = strategy
        self.product  = product

    def run(
        self,
        symbol: str,
        bars:   pd.DataFrame,
        fold_k: int = 0,
    ) -> BacktestResult:
        self.strategy.reset()

        open_trade: Optional[Trade] = None
        trades:     List[Trade]     = []
        equity_curve: List[float]   = []
        running_pnl = 0.0

        bar_list: List[tuple[pd.Timestamp, pd.Series]] = list(bars.iterrows())
        if not bar_list:
            return BacktestResult(
                symbol=symbol, fold=fold_k, n_bars=0, n_trades=0,
                gross_pnl=0.0, net_pnl=0.0, total_costs=0.0,
                win_rate=0.0, profit_factor=0.0, sharpe=0.0, max_dd=0.0,
            )

        pending: Optional[Signal] = None
        for i, (ts, bar) in enumerate(bar_list):
            # 1. Execute pending signal at current bar's open (next bar from signal POV)
            if pending is not None:
                fill_px = float(bar["open"])
                if pending.action == "BUY":
                    cost_in  = leg_cost(fill_px, pending.qty, "BUY", self.product)  # type: ignore[arg-type]
                    open_trade = Trade(
                        entry_ts=ts, exit_ts=None, symbol=symbol,
                        entry_px=fill_px, exit_px=None, qty=pending.qty,
                        cost=cost_in.total, reason=pending.reason,
                    )
                elif pending.action == "SELL" and open_trade is not None:
                    cost_out = leg_cost(fill_px, open_trade.qty, "SELL", self.product)  # type: ignore[arg-type]
                    open_trade.exit_ts   = ts
                    open_trade.exit_px   = fill_px
                    open_trade.gross_pnl = (fill_px - open_trade.entry_px) * open_trade.qty
                    open_trade.cost     += cost_out.total
                    open_trade.net_pnl   = open_trade.gross_pnl - open_trade.cost
                    running_pnl         += open_trade.net_pnl
                    trades.append(open_trade)
                    _notify_trade_closed(self.strategy, open_trade)
                    open_trade = None
                pending = None

            # 2. Feed the bar to the strategy
            signal = self.strategy.on_bar(ts, symbol, bar)
            if signal and signal.action in ("BUY", "SELL"):
                pending = signal

            # 3. Mark equity curve using close
            mark = running_pnl
            if open_trade is not None:
                mark += (float(bar["close"]) - open_trade.entry_px) * open_trade.qty
            equity_curve.append(mark)

        # End-of-fold: force-close any open position at the last bar's close
        if open_trade is not None:
            last_ts, last_bar = bar_list[-1]
            fill_px = float(last_bar["close"])
            cost_out = leg_cost(fill_px, open_trade.qty, "SELL", self.product)  # type: ignore[arg-type]
            open_trade.exit_ts   = last_ts
            open_trade.exit_px   = fill_px
            open_trade.gross_pnl = (fill_px - open_trade.entry_px) * open_trade.qty
            open_trade.cost     += cost_out.total
            open_trade.net_pnl   = open_trade.gross_pnl - open_trade.cost
            trades.append(open_trade)
            _notify_trade_closed(self.strategy, open_trade)

        return _build_result(symbol, fold_k, bars, trades, equity_curve)


def _notify_trade_closed(strategy: Strategy, trade: Trade) -> None:
    """Forward a closed trade to the strategy if it opted into the hook."""
    cb = getattr(strategy, "on_trade_closed", None)
    if cb is None or trade.exit_ts is None:
        return
    try:
        exit_dt = trade.exit_ts.to_pydatetime() if isinstance(trade.exit_ts, pd.Timestamp) else trade.exit_ts
        cb(trade.symbol, float(trade.net_pnl), exit_dt)
    except Exception as exc:
        logger.debug("on_trade_closed hook failed: %s", exc)


def _build_result(
    symbol:       str,
    fold_k:       int,
    bars:         pd.DataFrame,
    trades:       List[Trade],
    equity_curve: Sequence[float],
) -> BacktestResult:
    n_trades  = len(trades)
    gross     = sum(t.gross_pnl for t in trades)
    costs     = sum(t.cost      for t in trades)
    net       = sum(t.net_pnl   for t in trades)
    wins      = sum(1 for t in trades if t.net_pnl > 0)
    win_rate  = wins / n_trades if n_trades else 0.0
    # Profit factor = gross wins / |gross losses|. Inf if no losers. 0 if no winners.
    gross_wins   = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_losses = sum(-t.net_pnl for t in trades if t.net_pnl < 0)
    if gross_losses > 0:
        profit_factor = round(gross_wins / gross_losses, 3)
    elif gross_wins > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    # Sharpe from per-bar equity returns (annualised for ~252 × 375 minute bars)
    if len(equity_curve) > 1:
        series = pd.Series(equity_curve)
        rets   = series.diff().dropna()
        if rets.std() > 0:
            sharpe = float(rets.mean() / rets.std() * math.sqrt(252 * 375))
        else:
            sharpe = 0.0
        # Max drawdown in rupees
        peak     = series.cummax()
        drawdown = peak - series
        max_dd   = float(drawdown.max())
    else:
        sharpe = 0.0
        max_dd = 0.0

    return BacktestResult(
        symbol=symbol,
        fold=fold_k,
        n_bars=len(bars),
        n_trades=n_trades,
        gross_pnl=round(gross, 2),
        net_pnl=round(net, 2),
        total_costs=round(costs, 2),
        win_rate=round(win_rate, 4),
        profit_factor=profit_factor,
        sharpe=round(sharpe, 3),
        max_dd=round(max_dd, 2),
        trades=trades,
    )


# ---------------------------------------------------------------------------
# Convenience: run a walk-forward over one symbol
# ---------------------------------------------------------------------------
def run_walk_forward(
    symbol:     str,
    bars:       pd.DataFrame,
    strategy_factory: Callable[[], Strategy],
    splitter:   Optional[PurgedWalkForward] = None,
    product:    str = "MIS",
) -> List[BacktestResult]:
    splitter = splitter or PurgedWalkForward(n_splits=5, embargo_pct=0.02)
    folds    = splitter.split(len(bars))
    results: List[BacktestResult] = []
    for fold in folds:
        test_slice = bars.iloc[fold.test_start : fold.test_end]
        engine = BacktestEngine(strategy=strategy_factory(), product=product)
        res = engine.run(symbol=symbol, bars=test_slice, fold_k=fold.k)
        results.append(res)
        pf = "∞" if math.isinf(res.profit_factor) else f"{res.profit_factor:.2f}"
        logger.info(
            "Fold %d  bars=%d  trades=%d  net=₹%.2f  sharpe=%.2f  "
            "dd=₹%.2f  win_rate=%.1f%%  pf=%s",
            fold.k, res.n_bars, res.n_trades, res.net_pnl,
            res.sharpe, res.max_dd, res.win_rate * 100, pf,
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SentiStack walk-forward backtester")
    p.add_argument(
        "--symbol",
        default="RELIANCE",
        help="Single symbol OR comma-separated list (e.g. RELIANCE,TCS,INFY)",
    )
    p.add_argument("--start",     default="2025-01-06", help="YYYY-MM-DD (UTC)")
    p.add_argument("--end",       default="2025-01-31", help="YYYY-MM-DD (UTC)")
    p.add_argument("--interval",  default="minute")
    p.add_argument(
        "--strategy",
        choices=["flat", "momentum", "live_risk"],
        default="momentum",
        help="flat/momentum = scaffold; live_risk = real RiskManager adapter",
    )
    p.add_argument("--capital", type=float, default=100_000.0,
                   help="Starting capital for the live_risk adapter (₹)")
    p.add_argument("--n-splits",  type=int, default=5)
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic data instead of Zerodha history (offline mode)",
    )
    p.add_argument("--base-price", type=float, default=2500.0)
    p.add_argument(
        "--report-csv",
        default="backtest/reports/fold_summary.csv",
        help="Write per-fold summary to this CSV",
    )
    return p.parse_args(argv)


def _strategy_factory(kind: str, capital: float = 100_000.0) -> Callable[[], Strategy]:
    if kind == "flat":
        return lambda: DummyFlatStrategy()
    if kind == "momentum":
        return lambda: ScaffoldMomentumStrategy()
    if kind == "live_risk":
        # Late import so the base scaffold doesn't drag in the live
        # RiskManager / GRI / config stack unless the user asks for it.
        from .strategy_adapter import BacktestStrategyAdapter
        return lambda: BacktestStrategyAdapter(capital=capital)
    raise ValueError(kind)


def _load_or_synth(
    symbol:    str,
    start:     datetime,
    end:       datetime,
    interval:  str,
    synthetic: bool,
    base_price: float,
) -> pd.DataFrame:
    if synthetic:
        return generate_synthetic_bars(symbol, start, end, interval=interval, base_price=base_price)
    bars = load_bars(symbol, start, end, interval=interval)
    if bars.empty:
        logger.warning("No bars from live loader for %s — falling back to synthetic", symbol)
        bars = generate_synthetic_bars(symbol, start, end, interval=interval, base_price=base_price)
    return bars


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    symbols = [s.strip().upper() for s in args.symbol.split(",") if s.strip()]
    if not symbols:
        logger.error("No symbols supplied")
        return 1

    splitter = PurgedWalkForward(n_splits=args.n_splits, embargo_pct=0.02)
    factory  = _strategy_factory(args.strategy, capital=args.capital)

    all_results: List[BacktestResult] = []
    per_symbol_bars: dict[str, int]   = {}

    for sym in symbols:
        bars = _load_or_synth(sym, start, end, args.interval, args.synthetic, args.base_price)
        if bars.empty:
            logger.error("No bars for %s — skipping", sym)
            continue
        logger.info("Loaded %d bars for %s (%s → %s)", len(bars), sym, start.date(), end.date())
        per_symbol_bars[sym] = len(bars)
        results = run_walk_forward(
            symbol=sym,
            bars=bars,
            strategy_factory=factory,
            splitter=splitter,
            product="MIS",
        )
        all_results.extend(results)

    if not all_results:
        logger.error("No results produced — aborting")
        return 1

    # Aggregate summary
    total_trades = sum(r.n_trades for r in all_results)
    total_net    = sum(r.net_pnl  for r in all_results)
    total_cost   = sum(r.total_costs for r in all_results)
    total_gross  = sum(r.gross_pnl for r in all_results)
    gross_wins   = sum(r.net_pnl for r in all_results if r.net_pnl > 0)
    gross_losses = sum(-r.net_pnl for r in all_results if r.net_pnl < 0)
    if gross_losses > 0:
        agg_pf = f"{gross_wins / gross_losses:.2f}"
    elif gross_wins > 0:
        agg_pf = "∞"
    else:
        agg_pf = "0.00"

    total_bars = sum(per_symbol_bars.values())
    print("\n" + "=" * 64)
    print(
        f"Backtest summary — {len(symbols)} symbol(s) "
        f"[{args.strategy} strategy, {args.n_splits} folds]"
    )
    print("=" * 64)
    print(f"  symbols:        {','.join(symbols)}")
    print(f"  total bars:     {total_bars:>12,}")
    print(f"  total trades:   {total_trades:>12,}")
    print(f"  gross P&L:      ₹{total_gross:>12,.2f}")
    print(f"  total costs:    ₹{total_cost:>12,.2f}")
    print(f"  net P&L:        ₹{total_net:>12,.2f}")
    print(f"  profit factor:  {agg_pf:>12}")
    if all_results:
        avg_sharpe = sum(r.sharpe for r in all_results) / len(all_results)
        print(f"  avg sharpe:     {avg_sharpe:>12.3f}")
    print("=" * 64)

    # Write CSV report
    from pathlib import Path
    report_path = Path(args.report_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.as_dict() for r in all_results])
    df.to_csv(report_path, index=False)
    print(f"Per-fold summary → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
