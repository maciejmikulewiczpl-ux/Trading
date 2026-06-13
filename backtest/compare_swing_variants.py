"""compare_swing_variants.py -- gates G0-G4 for the swing-engine backtest.

Runs V0 / V1 / V2 from run_swing.py, prints full output per spec S4, then
evaluates all five pre-registered gates (SWING_ENGINE_SPEC.md S5).

G3 (decorrelation vs ORB): reconstructs ORB daily PnL from the existing
.bars_cache_trades_730d.pkl (tight-OR<=0.5%, all longs, exit-date bucketed).

Run:
    .venv-openbb\\Scripts\\python.exe backtest\\compare_swing_variants.py
(cache must exist: run fetch_swing_data.py first)
"""
from __future__ import annotations

import math
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_swing import run_simulation, SwingTrade, CACHE, SIM_START  # noqa: E402

HALF_SPLIT = date(2021, 1, 1)
CRISIS_2020 = (date(2020, 2, 1), date(2020, 4, 30))
CRISIS_2022 = (date(2022, 1, 1), date(2022, 12, 31))
CRISIS_2025 = (date(2025, 4, 1), date(2025, 4, 30))


# ---------------------------------------------------------------- stats helpers

def sharpe(daily: pd.Series) -> float:
    if daily.std() == 0 or len(daily) < 10:
        return 0.0
    return float(daily.mean() / daily.std() * math.sqrt(252))


def max_drawdown(daily: pd.Series) -> float:
    cum = daily.cumsum()
    return float((cum - cum.cummax()).min())


def window_pnl(trades: list[SwingTrade], d0: date, d1: date) -> float:
    return sum(t.pnl_net for t in trades if d0 <= t.exit_date <= d1)


def window_daily_pnl(daily: pd.Series, d0: date, d1: date) -> pd.Series:
    return daily[(daily.index >= d0) & (daily.index <= d1)]


def pnl_by_year(trades: list[SwingTrade]) -> dict[int, float]:
    by: dict[int, float] = {}
    for t in trades:
        by[t.exit_date.year] = by.get(t.exit_date.year, 0.0) + t.pnl_net
    return by


# ---------------------------------------------------------------- ORB daily PnL (G3)

def orb_daily_pnl() -> pd.Series:
    """Reconstruct ORB daily PnL from cached trades (exit-date bucketed, tight-OR<=0.5%).
    Avoids alpaca import chain: uses only stdlib + pickle + pandas."""
    try:
        trades = pickle.load(open(ROOT / "backtest" / ".bars_cache_trades_730d.pkl", "rb"))
    except FileNotFoundError:
        return pd.Series(dtype=float)

    def _or_pct(t) -> float:
        try:
            return (t.or_high - t.or_low) / t.entry_price * 100
        except Exception:
            return 999.0

    def _tday(t) -> date:
        try:
            return t.entry_time.date()
        except Exception:
            return t.exit_time.date()

    tight = [t for t in trades if getattr(t, "side", "long") == "long" and _or_pct(t) <= 0.5]
    by: dict[date, float] = {}
    for t in tight:
        d = _tday(t)
        by[d] = by.get(d, 0.0) + t.pnl_dollars
    return pd.Series(by).sort_index()


# ---------------------------------------------------------------- print helpers

def print_trades(trades: list[SwingTrade], label: str) -> None:
    print(f"\n  {label} -- per-trade list (first 30 and last 10 shown):")
    header = f"  {'sym':<6} {'entry':<12} {'exit':<12} {'hold':>4} {'entry_px':>8} {'exit_px':>8} {'R_net':>8}  reason"
    print(header)
    show = trades[:30] + (trades[-10:] if len(trades) > 40 else [])
    for t in show:
        r = t.pnl_net / (t.entry_price * t.shares * 0.05) if t.entry_price > 0 else 0
        print(f"  {t.symbol:<6} {t.entry_date!s:<12} {t.exit_date!s:<12} {t.hold_days:>4} "
              f"{t.entry_price:>8.2f} {t.exit_price:>8.2f} {t.pnl_net:>+8.0f}  {t.exit_reason}")


def print_summary(label: str, trades: list[SwingTrade], daily: pd.Series,
                  spy_ret: float, orb_ser: pd.Series) -> dict:
    wins = [t for t in trades if t.pnl_net > 0]
    losses = [t for t in trades if t.pnl_net <= 0]
    hold_days = [t.hold_days for t in trades]
    avg_win = sum(t.pnl_net for t in wins) / max(len(wins), 1)
    avg_loss = sum(t.pnl_net for t in losses) / max(len(losses), 1)
    win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    yr = pnl_by_year(trades)

    print(f"\n{'='*70}")
    print(f"  VARIANT: {label}")
    print(f"{'='*70}")
    print(f"  Trades: {len(trades)}  |  Wins: {len(wins)}  Losses: {len(losses)}  "
          f"Win rate: {100*len(wins)/max(len(trades),1):.0f}%")
    print(f"  Net PnL: ${sum(t.pnl_net for t in trades):+,.0f}  |  "
          f"Avg win: ${avg_win:+,.0f}  Avg loss: ${avg_loss:+,.0f}  "
          f"Win/loss ratio: {win_loss:.2f}x")
    print(f"  Sharpe (annualised daily MTM): {sharpe(daily):.2f}")
    print(f"  Max drawdown: ${max_drawdown(daily):,.0f}")
    print(f"  Avg hold: {sum(hold_days)/max(len(hold_days),1):.1f}d  "
          f"Median hold: {sorted(hold_days)[len(hold_days)//2] if hold_days else 0}d")
    avg_open = daily.count()  # proxy; actual avg open computed below
    # SPY benchmark
    print(f"  SPY buy-hold same window: {spy_ret:+.1f}%")

    print(f"\n  PnL by year:")
    for y in sorted(yr):
        print(f"    {y}: ${yr[y]:+,.0f}")

    for label_w, d0, d1 in (
        ("2020-02->04 (COVID crash)", *CRISIS_2020),
        ("2022 (bear year)", *CRISIS_2022),
        ("2025-04 (tariff spike)", *CRISIS_2025),
    ):
        w = window_pnl(trades, d0, d1)
        wd = window_daily_pnl(daily, d0, d1)
        print(f"  {label_w}: PnL ${w:+,.0f}  Sharpe {sharpe(wd):.2f}")

    # halves
    h1_trades = [t for t in trades if t.exit_date < HALF_SPLIT]
    h2_trades = [t for t in trades if t.exit_date >= HALF_SPLIT]
    h1_pnl = sum(t.pnl_net for t in h1_trades)
    h2_pnl = sum(t.pnl_net for t in h2_trades)
    print(f"\n  H1 (2016-2020): trades={len(h1_trades)}  PnL=${h1_pnl:+,.0f}")
    print(f"  H2 (2021-2026): trades={len(h2_trades)}  PnL=${h2_pnl:+,.0f}")

    # G3 correlation
    corr_val = float("nan")
    if len(orb_ser) > 0:
        overlap_idx = daily.index[daily.index.isin(orb_ser.index)]
        if len(overlap_idx) >= 30:
            corr_val = float(daily[overlap_idx].corr(orb_ser[overlap_idx]))
    print(f"\n  G3 ORB correlation over 730d overlap: {corr_val:.3f}  (gate: <= 0.30)")

    return {
        "n": len(trades), "sharpe": sharpe(daily), "maxdd": max_drawdown(daily),
        "pnl": sum(t.pnl_net for t in trades),
        "h1_pnl": h1_pnl, "h2_pnl": h2_pnl,
        "median_hold": sorted(hold_days)[len(hold_days)//2] if hold_days else 0,
        "win_loss_ratio": win_loss, "corr_orb": corr_val,
        "crisis_2020": window_pnl(trades, *CRISIS_2020),
        "crisis_2022": window_pnl(trades, *CRISIS_2022),
    }


def gate_row(name: str, value: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  {name:<12} {value:<40} {status}")


def print_gates(label: str, stats: dict, data: dict) -> bool:
    spy_close = data["spy"]["Close"]
    spy_sub2 = spy_close[spy_close.index.year >= SIM_START.year]
    spy_ret_full = float(float(spy_sub2.iloc[-1]) / float(spy_sub2.iloc[0]) - 1) * 100

    print(f"\n{'='*70}")
    print(f"  GATE TABLE -- {label}")
    print(f"{'='*70}")

    g0 = stats["n"] >= 150
    g1a = stats["sharpe"] >= 1.0
    g1b = stats["h1_pnl"] > 0 and stats["h2_pnl"] > 0
    g2a = stats["crisis_2022"] >= -500
    g2b = stats["crisis_2020"] >= -1000
    g3 = not math.isnan(stats["corr_orb"]) and stats["corr_orb"] <= 0.30
    g4a = stats["median_hold"] >= 3
    g4b = stats["win_loss_ratio"] >= 1.8

    gate_row("G0 validity", f">= 150 trades: {stats['n']}", g0)
    gate_row("G1a econ", f"Sharpe >= 1.0: {stats['sharpe']:.2f}", g1a)
    gate_row("G1b halves", f"both halves positive: H1=${stats['h1_pnl']:+,.0f} H2=${stats['h2_pnl']:+,.0f}", g1b)
    gate_row("G2a 2022", f">= -$500: ${stats['crisis_2022']:+,.0f}", g2a)
    gate_row("G2b 2020", f">= -$1000: ${stats['crisis_2020']:+,.0f}", g2b)
    gate_row("G3 decorr", f"<= 0.30: {stats['corr_orb']:.3f}", g3)
    gate_row("G4a hold", f"median >= 3d: {stats['median_hold']}d", g4a)
    gate_row("G4b ratio", f"win/loss >= 1.8x: {stats['win_loss_ratio']:.2f}x", g4b)

    all_pass = all([g0, g1a, g1b, g2a, g2b, g3, g4a, g4b])
    verdict = "ALL GATES PASS -- proceed to robustness sweep" if all_pass \
        else "GATE(S) FAILED -- variant does not qualify"
    print(f"\n  >> {verdict}")
    return all_pass


def main() -> None:
    print("Loading cache...")
    data = pickle.load(open(CACHE, "rb"))
    spy_close = data["spy"]["Close"]
    spy_sub = spy_close[spy_close.index.year >= 2016]
    spy_ret = float(float(spy_sub.iloc[-1]) / float(spy_sub.iloc[0]) - 1) * 100
    print(f"Universe: {len(data['symbols'])} symbols | "
          f"SPY period return: {spy_ret:+.1f}%")

    orb_ser = orb_daily_pnl()
    print(f"ORB daily series: {len(orb_ser)} sessions "
          f"({orb_ser.index[0] if len(orb_ser) else 'n/a'} -> "
          f"{orb_ser.index[-1] if len(orb_ser) else 'n/a'})")

    results = {}
    for label, variant in (("V0 (55d Donchian)", "V0"),
                           ("V1 (+ compression)", "V1"),
                           ("V2 (20d horizon)", "V2")):
        print(f"\nRunning {label}...")
        trades, daily = run_simulation(data, variant=variant)
        print(f"  -> {len(trades)} closed trades")
        stats = print_summary(label, trades, daily, spy_ret, orb_ser)
        passed = print_gates(label, stats, data)
        results[variant] = (trades, daily, stats, passed)

    # candidate for robustness: V1 if it passes, else V0 if it passes
    candidate = None
    for v in ("V1", "V0", "V2"):
        if results[v][3]:
            candidate = v
            break

    print(f"\n{'='*70}")
    if candidate:
        print(f"  CANDIDATE FOR ROBUSTNESS SWEEP: {candidate}")
        print(f"  Run: .venv-openbb\\Scripts\\python.exe backtest\\swing_robustness.py --variant {candidate}")
    else:
        print(f"  NO VARIANT PASSED ALL GATES -- swing engine REJECTED")
        print(f"  Record verdict and close the cell per spec discipline.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
