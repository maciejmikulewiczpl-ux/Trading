"""run_swing.py -- day-by-day swing-breakout simulator (Engine #3 spec).

Implements V0 (55d Donchian + chandelier), V1 (+ compression precondition),
V2 (20d entry horizon). Called by compare_swing_variants.py and
swing_robustness.py; not meant to run standalone.

All rules follow SWING_ENGINE_SPEC.md verbatim. No tuning.
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "backtest" / ".swing_daily_cache.pkl"

# Inlined from pit_expand to avoid alpaca import chain in .venv-openbb
BLOCK = {"TQQQ", "SQQQ", "SOXL", "SOXS", "TZA", "TNA", "SPXL", "SPXS", "UPRO",
         "UVXY", "SVXY", "TMF", "TMV", "YINN", "FNGU", "BOIL", "UCO",
         "MSTR", "IBIT", "ETHA", "BITO", "BMNR", "CRCL", "CRWV", "MARA", "RIOT"}
ETFS = {"EEM", "EFA", "EWY", "EWZ", "FXI", "GDX", "GLD", "IEFA", "IVV", "KRE", "KWEB",
        "RSP", "SLV", "SMH", "SOXX", "XBI", "XLE", "XLF", "XLI", "XLK", "XLU", "XLV",
        "AGG", "TLT", "IEF", "LQD", "HYG", "JNK", "EMB", "VCIT", "VCLT", "USHY",
        "SPY", "QQQ", "IWM", "DIA", "VOO", "IGV"}
EXCLUDE = BLOCK | ETFS | {"SPY"}

# ---- portfolio params (from spec) ---
RISK_PER_TRADE = 50.0
MAX_SLOTS = 12
MAX_NOTIONAL = 10_000.0
MIN_STOP_DIST = 0.05
COST_RT_PCT = 0.001     # 0.10% round-trip of entry notional

SIM_START = date(2016, 1, 1)   # first signal date
COMPRESSION_PCTILE = 33        # pct for V1 range compression
ATR_PERIOD = 14
RANGE_PERIOD = 10
PCTILE_WINDOW = 252


# ------------------------------------------------------------------ helpers

def wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Wilder ATR using EWM (com = period-1, alpha = 1/period)."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def rolling_high(series: pd.Series, window: int) -> pd.Series:
    """Rolling max of prior `window` periods (no lookahead — shift(1) before rolling)."""
    return series.shift(1).rolling(window, min_periods=window).max()


def range_pct_series(df: pd.DataFrame, w: int = RANGE_PERIOD) -> pd.Series:
    """(rolling_high_w - rolling_low_w) / close -- uses data up to and including today."""
    rh = df["High"].rolling(w, min_periods=w).max()
    rl = df["Low"].rolling(w, min_periods=w).min()
    return (rh - rl) / df["Close"].replace(0, np.nan)


@dataclass
class SwingTrade:
    symbol: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    shares: int
    hold_days: int
    exit_reason: str
    pnl_gross: float = 0.0
    cost: float = 0.0
    pnl_net: float = 0.0

    def __post_init__(self):
        self.pnl_gross = (self.exit_price - self.entry_price) * self.shares
        self.cost = COST_RT_PCT * self.entry_price * self.shares
        self.pnl_net = self.pnl_gross - self.cost


# ---------------------------------------------------------------- precompute

def precompute(data: dict, entry_days: int, use_compression: bool
               ) -> dict[str, pd.DataFrame]:
    """Per-symbol indicator DataFrame (all lookahead-free)."""
    out = {}
    for sym, df in data["symbols"].items():
        if sym in EXCLUDE:
            continue
        df = df.copy()
        # ATR (Wilder) -- strictly trailing (prev day values used in sim)
        df["atr"] = wilder_atr(df)
        # rolling high of prior N sessions (no today)
        df["rh55"] = rolling_high(df["High"], 55)
        df["rh20"] = rolling_high(df["High"], 20)
        # range compression (includes today's bar -- computed at EOD of signal day)
        if use_compression:
            rp = range_pct_series(df)
            # 33rd pctile of rp over trailing 252 sessions (no lookahead: shift+rolling)
            df["range_pct"] = rp
            df["range_pct_33p"] = rp.shift(1).rolling(PCTILE_WINDOW, min_periods=60).quantile(
                COMPRESSION_PCTILE / 100.0
            )
        # dollar-vol for entry ranking (20d mean, no lookahead)
        df["dvol20"] = (df["Close"] * df["Volume"]).shift(1).rolling(20, min_periods=5).mean()
        out[sym] = df
    return out


def build_yearly_universes(data: dict, indicators: dict[str, pd.DataFrame]
                           ) -> dict[int, set]:
    """Top 100 symbols by prior-year mean dvol, refreshed each year."""
    spy = data["spy"]
    years = range(SIM_START.year, spy.index[-1].year + 2)
    yearly: dict[int, set] = {}
    for yr in years:
        # reference: trading days in year yr-1
        ref_idx = spy.index[spy.index.year == (yr - 1)]
        if len(ref_idx) == 0:
            yearly[yr] = set()
            continue
        dvols: dict[str, float] = {}
        for sym, df in indicators.items():
            if sym in EXCLUDE:
                continue
            sub = df.loc[df.index.isin(ref_idx), "dvol20"].dropna()
            if len(sub) < 30:
                continue
            dvols[sym] = float(sub.mean())
        top = sorted(dvols, key=dvols.__getitem__, reverse=True)[:100]
        yearly[yr] = set(top)
    return yearly


# ---------------------------------------------------------------- simulation

def run_simulation(
    data: dict,
    variant: str = "V0",
    entry_days: int = 55,
    stop_mult: float = 2.5,
    use_compression: Optional[bool] = None,
) -> tuple[list[SwingTrade], pd.Series]:
    """
    Returns (trades, daily_pnl_series).
    daily_pnl indexed by date, values = portfolio daily MTM change.
    """
    if use_compression is None:
        use_compression = (variant == "V1")
    if variant == "V2":
        entry_days = 20

    rh_col = "rh55" if entry_days == 55 else "rh20"

    ind = precompute(data, entry_days, use_compression)
    yearly_uni = build_yearly_universes(data, ind)

    spy = data["spy"]
    spy["sma200"] = spy["Close"].shift(1).rolling(200, min_periods=150).mean()
    spy_idx = pd.DatetimeIndex(spy.index)

    # trading days from SIM_START onward
    trading_days = [
        d.date() for d in spy_idx if d.date() >= SIM_START
    ]

    # position dict: sym -> {entry_price, shares, max_close, initial_stop, entry_date}
    positions: dict = {}
    trades: list[SwingTrade] = []
    daily_pnl: dict[date, float] = {}

    for i, D in enumerate(trading_days):
        ts_D = pd.Timestamp(D)
        ts_prev = pd.Timestamp(trading_days[i - 1]) if i > 0 else None

        spy_row = spy.loc[ts_D] if ts_D in spy.index else None
        if spy_row is None:
            continue

        # regime: SPY close(T-1) > SMA200(T-1) -- lookahead-free
        spy_regime = False
        if ts_prev is not None and ts_prev in spy.index:
            prev_spy = spy.loc[ts_prev]
            if pd.notna(prev_spy["Close"]) and pd.notna(prev_spy["sma200"]):
                spy_regime = float(prev_spy["Close"]) > float(prev_spy["sma200"])

        day_realized = 0.0
        exits_today = []

        # ---- 1. Exit check -----------------------------------------------
        for sym in list(positions.keys()):
            pos = positions[sym]
            df = ind.get(sym)
            if df is None or ts_prev is None or ts_prev not in df.index:
                continue
            if ts_D not in df.index:
                continue

            row_D = df.loc[ts_D]
            row_prev = df.loc[ts_prev]
            atr_prev = float(row_prev["atr"]) if pd.notna(row_prev["atr"]) else None
            if atr_prev is None or atr_prev <= 0:
                continue

            # chandelier stop = max_close - mult * ATR (from prev day's data)
            chandelier = float(pos["max_close"]) - stop_mult * atr_prev
            stop_level = max(chandelier, pos["initial_stop"])

            o = float(row_D["Open"]) if pd.notna(row_D["Open"]) else None
            lo = float(row_D["Low"]) if pd.notna(row_D["Low"]) else None
            c = float(row_D["Close"]) if pd.notna(row_D["Close"]) else None

            if o is None or lo is None or c is None:
                continue

            if o <= stop_level:
                # gap-through: fill at open
                exit_price = o
                reason = "gap_stop"
            elif lo <= stop_level:
                exit_price = stop_level
                reason = "stop"
            else:
                # update max_close
                pos["max_close"] = max(pos["max_close"], c)
                continue

            hold = (D - pos["entry_date"]).days
            t = SwingTrade(
                symbol=sym, entry_date=pos["entry_date"], exit_date=D,
                entry_price=pos["entry_price"], exit_price=exit_price,
                shares=pos["shares"], hold_days=hold, exit_reason=reason,
            )
            trades.append(t)
            exits_today.append(t.pnl_net)
            day_realized += t.pnl_net
            del positions[sym]

        # ---- 2. MTM for remaining open positions -------------------------
        mtm = 0.0
        if ts_prev is not None:
            for sym, pos in positions.items():
                df = ind.get(sym)
                if df is None:
                    continue
                if ts_D in df.index and ts_prev in df.index:
                    c_today = df.loc[ts_D, "Close"]
                    c_prev = df.loc[ts_prev, "Close"]
                    if pd.notna(c_today) and pd.notna(c_prev):
                        mtm += (float(c_today) - float(c_prev)) * pos["shares"]

        daily_pnl[D] = day_realized + mtm

        # ---- 3. New entries (signal from yesterday, entry today at open) -
        if len(positions) >= MAX_SLOTS:
            continue

        uni = yearly_uni.get(D.year, set())
        if not uni:
            continue

        signals = []
        for sym in uni:
            if sym not in ind:
                continue
            if sym in positions:
                continue
            df = ind[sym]
            if ts_prev not in df.index:
                continue
            row_prev = df.loc[ts_prev]

            # entry signal: close(T-1) > rolling high of prior entry_days
            c_prev = row_prev.get("Close", float("nan"))
            rh_prev = row_prev.get(rh_col, float("nan"))
            if pd.isna(c_prev) or pd.isna(rh_prev):
                continue
            if float(c_prev) <= float(rh_prev):
                continue

            # compression precondition (V1)
            if use_compression:
                rp = row_prev.get("range_pct", float("nan"))
                rp33 = row_prev.get("range_pct_33p", float("nan"))
                if pd.isna(rp) or pd.isna(rp33):
                    continue
                if float(rp) > float(rp33):
                    continue

            # regime gate
            if not spy_regime:
                continue

            dvol = float(row_prev["dvol20"]) if pd.notna(row_prev.get("dvol20")) else 0.0
            signals.append((sym, dvol))

        # sort by dollar-vol descending
        signals.sort(key=lambda x: x[1], reverse=True)

        for sym, _ in signals:
            if len(positions) >= MAX_SLOTS:
                break
            df = ind[sym]
            if ts_D not in df.index:
                continue
            row_D = df.loc[ts_D]
            row_prev = df.loc[ts_prev]

            o = float(row_D["Open"]) if pd.notna(row_D["Open"]) else None
            if o is None or o <= 0:
                continue

            atr_prev = float(row_prev["atr"]) if pd.notna(row_prev["atr"]) else None
            if atr_prev is None or atr_prev <= 0:
                continue

            initial_stop = o - stop_mult * atr_prev
            stop_dist = o - initial_stop
            if stop_dist < MIN_STOP_DIST:
                continue

            # gap-collapsed: if open already below initial_stop, skip
            if o < initial_stop:
                continue

            shares = min(
                int(math.floor(RISK_PER_TRADE / stop_dist)),
                int(math.floor(MAX_NOTIONAL / o)),
            )
            if shares <= 0:
                continue

            c_today = float(row_D["Close"]) if pd.notna(row_D["Close"]) else o
            positions[sym] = {
                "entry_price": o,
                "shares": shares,
                "max_close": c_today,
                "initial_stop": initial_stop,
                "entry_date": D,
            }

    # close any still-open positions at last day's close (mark-only, not a real exit)
    # -- excluded from trade stats per spec (only closed trades count)

    daily_ser = pd.Series(daily_pnl).sort_index()
    return trades, daily_ser
