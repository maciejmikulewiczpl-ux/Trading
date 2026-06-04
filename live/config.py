"""Editable ORB parameters: schema, defaults, load/save, and builders.

Single source of truth shared by the live runner (live/paper_orb.py) and the
settings GUI (live/config_ui.py). Defaults below are the values validated across
this project's backtests (see each Setting.help for the basis). If
live/orb_config.json is absent or unreadable, the live runner uses these exact
defaults — so adding/removing the file never changes behavior unexpectedly.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "live" / "orb_config.json"

log = logging.getLogger("orb_paper")


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    group: str
    kind: str  # int | float | optfloat | bool | time | csv | str
    default: object
    help: str
    minv: Optional[float] = None
    maxv: Optional[float] = None


# Order here is the display order in the GUI; `group` controls section grouping.
SETTINGS: list[Setting] = [
    # ---- Universe ----
    Setting(
        "watchlist", "Watchlist (symbols)", "Universe", "csv",
        [
            # ETFs (4)
            "SPY", "QQQ", "IWM", "DIA",
            # Mega/large-cap tech & semis (21)
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL",
            "AMD", "NFLX", "ADBE", "CRM", "INTC", "CSCO", "QCOM", "TXN", "MU",
            "ACN", "IBM", "NOW", "PANW", "CRWD", "SHOP", "ABNB", "INTU", "PYPL",
            "ANET", "AMAT", "LRCX",
            # Financials (15)
            "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP",
            "SCHW", "BLK", "USB", "SPGI", "ICE", "CME",
            # Healthcare (14)
            "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV",
            "TMO", "ABT", "DHR", "AMGN", "ISRG", "BMY", "ELV", "CVS",
            # Consumer (17)
            "WMT", "HD", "COST", "NKE", "MCD", "SBUX", "DIS", "KO", "PEP",
            "TGT", "LOW", "BKNG", "LULU", "ROST", "CMG", "F", "GM",
            # Industrial / energy (11)
            "XOM", "CVX", "CAT", "BA", "GE",
            "HON", "RTX", "LMT", "DE", "UNP", "COP",
            # Telecom / utility (5)
            "T", "VZ", "TMUS", "CMCSA", "NEE",
            # Higher-beta / high-volume (4)
            "PLTR", "COIN", "UBER", "BABA",
        ],
        "Symbols the bot trades for LONG breakouts. Comma-separated. Broad ~100 "
        "liquid US large-caps (roughly S&P-100 + a few momentum/high-volume "
        "names + 4 ETFs). universe_scan.py validated the breadth thesis (the "
        "ORB edge is OOS-robust only when averaged over many names; per-name "
        "selection is noise). Pair with the trend filter (200d SMA + RS) and "
        "the concurrency cap so the filter has many candidates to pick the "
        "strongest setups from. ~100 names is the practical ceiling before "
        "intraday fetches stretch past the 10s poll cycle.",
    ),

    # ---- Opening range / long entries ----
    Setting(
        "or_minutes", "Opening-range minutes", "Strategy", "int", 15,
        "Length of the opening range, measured from 9:30 ET. The OR high/low set "
        "the long breakout trigger and the stop. 15 min is the project default; "
        "the sweep (sweep_orb.py) did not find a materially better value.",
        minv=1, maxv=120,
    ),
    Setting(
        "target_r", "Target (R multiple)", "Strategy", "float", 2.0,
        "Profit target as a multiple of risk (entry-to-stop distance). 2.0R means "
        "the take-profit sits twice as far from entry as the stop. Backtest sweeps "
        "favoured ~2.0R as a robust choice across the universe.",
        minv=0.25, maxv=10.0,
    ),
    Setting(
        "no_entry_after_time", "No new entries after (ET)", "Strategy", "time", "11:30",
        "Block NEW entries after this time (existing trades still ride to the EOD "
        "flatten). ORB breakouts after the first ~2h have weak follow-through. "
        "compare_cutoff.py showed 11:30 ET roughly doubles PnL vs no cutoff and is "
        "more robust than the (over-fit) 11:00 local optimum. Blank = no cutoff.",
    ),
    Setting(
        "move_stop_to_be_at_r", "Move stop to break-even at (R)", "Strategy", "optfloat", None,
        "If set, lift the stop to entry once price has moved this many R in your "
        "favour. Left OFF: compare_be_lift.py showed BE-lift at +0.5R and +1.0R "
        "both UNDERPERFORM no-lift here (it stops out trades that would finish "
        "positive at EOD). Blank = off.",
        minv=0.1, maxv=5.0,
    ),
    Setting(
        "stop_offset_pct", "Stop buffer (fraction of OR range)", "Strategy", "float", 0.0,
        "Push the stop this fraction of the OR range BEYOND the OR boundary "
        "(below OR-low for longs, above OR-high for shorts) to avoid liquidity "
        "sweeps at the round number. compare_stop_offset.py found no consistent "
        "improvement, so default is 0.0 (stop exactly at the OR boundary).",
        minv=0.0, maxv=0.5,
    ),
    Setting(
        "trend_filter_enabled", "Trend filter (200d SMA + RS)", "Strategy", "bool", True,
        "Only take LONG breakouts on names that, as of the prior trading day's "
        "close, satisfy BOTH: (1) close > 200-day SMA (daily uptrend) AND (2) "
        "20-day return > SPY's 20-day return (cross-sectional relative strength). "
        "Multi-timeframe momentum confirmation: compare_trend_filter.py shows the "
        "combo nearly DOUBLES avg_R (+0.045 -> +0.085), lifts win rate to 50.4%, "
        "and slashes max DD ~79% (-$11,687 -> -$2,412), while staying positive in "
        "BOTH OOS halves. Trade-off: ~62% fewer trades, ~38% lower raw PnL — same "
        "edge with a far smoother equity curve. Fails open (treats all eligible) "
        "if the daily fetch fails so it can never silently disable trading.",
    ),

    # ---- Sizing / risk guardrails ----
    Setting(
        "risk_per_trade", "Risk per trade ($)", "Risk", "float", 50.0,
        "Dollars risked per trade: shares = risk_per_trade / |entry - stop|. The "
        "core position-sizing knob. $50 pairs with 16 concurrent positions to hold "
        "the ~$800/day total risk budget while spreading it wider — diversification "
        "that ~2x'd Sharpe and cut drawdown ~30% vs the old 8x$100 "
        "(backtest/compare_capaware.py, 2026-06-04).",
        minv=1.0, maxv=100000.0,
    ),
    Setting(
        "max_position_pct", "Max position (% of equity)", "Risk", "float", 0.25,
        "Cap a single position at this fraction of account equity (0.25 = 25%). "
        "Combined with the dollar cap below; the smaller of the two wins.",
        minv=0.01, maxv=1.0,
    ),
    Setting(
        "max_position_dollars", "Max position notional ($)", "Risk", "optfloat", 10000.0,
        "Absolute cap on a single position's notional value, regardless of "
        "equity. $10,000 default keeps any one name from dominating. Blank = "
        "only the % cap applies.",
        minv=100.0, maxv=1000000.0,
    ),
    Setting(
        "daily_loss_cap", "Daily loss cap ($)", "Risk", "float", 500.0,
        "Circuit breaker: once realized PnL for the day is below -this amount, "
        "NEW entries halt (existing positions ride their brackets). $500 default.",
        minv=0.0, maxv=1000000.0,
    ),
    Setting(
        "max_concurrent_positions", "Max concurrent positions", "Risk", "int", 16,
        "Cap on how many positions may be open at once. With a broad watchlist many "
        "breakouts fire near the open; this stops $100k from being over-deployed. "
        "Raised 8->16 (with risk_per_trade $100->$50, same ~$800/day risk) because "
        "spreading the SAME risk across more names diversifies: ~2x Sharpe, ~30% less "
        "drawdown, ~2x PnL (the old cap-8 was throttled by the $10k notional cap) — "
        "backtest/compare_capaware.py 2026-06-04. When full, further breakouts are "
        "skipped for the day (first-come). 0 = unlimited.",
        minv=0, maxv=100,
    ),
    Setting(
        "trailing_exit_enabled", "Trailing-stop exit (let winners run)", "Risk", "bool", False,
        "Replace the fixed 2R take-profit target with a TRAILING stop: enter on a "
        "plain market order, then attach an Alpaca native trailing stop that trails "
        "1R (the initial entry-to-stop distance) below the high-water mark, with no "
        "fixed target — so trend-day winners run past 2R. backtest/compare_exits.py "
        "(+ compare_exits_slippage.py): ~2x avgR/Sharpe/PnL and ~half the drawdown "
        "vs fixed 2R, and the only net-positive config after measured slippage. OFF "
        "by default (it changes live exit execution); enable deliberately, watch the "
        "first session. Long-only; with shorts on it falls back to the bracket.",
    ),
    Setting(
        "min_risk_per_share", "Min risk/share ($)", "Risk", "float", 0.05,
        "Reject a setup if entry-to-stop is tighter than this — too tight means "
        "oversized share counts and noise-driven stops.",
        minv=0.0, maxv=100.0,
    ),
    Setting(
        "max_risk_per_share", "Max risk/share ($)", "Risk", "float", 10.00,
        "Reject a setup if entry-to-stop is wider than this — too wide means a "
        "huge OR range / unfavourable R math.",
        minv=0.0, maxv=1000.0,
    ),

    # ---- Short side (regime-gated) ----
    Setting(
        "short_enabled", "Enable shorts", "Shorts", "bool", False,
        "Master switch for the short side. Shorts are REGIME-GATED: a naive "
        "always-on short LOSES out-of-sample (validate_short_oos.py). STAGED OFF "
        "as of the 55-name watchlist rollout — we change one big thing at a time: "
        "validate that breadth (longs) helps live first, THEN re-enable shorts. "
        "Flip back to true once the broad long-only config has live data.",
    ),
    Setting(
        "short_symbols", "Short-eligible symbols", "Shorts", "csv",
        ["SPY", "QQQ", "NVDA", "AAPL"],
        "Which names may be shorted (subset of the watchlist). TSLA is excluded: "
        "per-symbol analysis showed single-name short squeezes (esp. TSLA) were "
        "the entire short loss, while index/large-cap shorts were profitable.",
    ),
    Setting(
        "regime_ref_symbol", "Regime reference symbol", "Shorts", "str", "SPY",
        "The broad-market proxy whose trend decides the short regime. SPY "
        "represents the overall US market.",
    ),
    Setting(
        "regime_sma_window", "Regime SMA window (days)", "Shorts", "int", 20,
        "Shorts are enabled only when the reference symbol's daily close is below "
        "its N-day simple moving average. 20 days (≈ one trading month) was the "
        "robust choice; with the confirmation filter below, windows 15-30 all "
        "beat long-only (regime_hardened.py).",
        minv=2, maxv=200,
    ),
    Setting(
        "regime_confirm_days", "Regime confirm days", "Shorts", "int", 3,
        "Require the reference symbol below its SMA for THIS many consecutive "
        "prior closes before enabling shorts. This 'confirmation' is what fixed "
        "the fragility: it kills one-day dips (the 2023 whipsaw) and makes the "
        "edge robust across SMA windows. N=3-5 all work; 3 is a good middle.",
        minv=1, maxv=15,
    ),
    Setting(
        "max_flips", "Max same-day flips", "Shorts", "int", 0,
        "Allow this many opposite-direction re-entries after a stop-out in the "
        "same session. Deployed at 0 (flip-free): backtest showed flip-free has "
        "HIGHER total PnL and ~baseline drawdown; the flip only slightly smooths "
        "drawdown while adding live state-machine complexity.",
        minv=0, maxv=3,
    ),
]

DEFAULTS: dict = {s.key: s.default for s in SETTINGS}
BY_KEY: dict[str, Setting] = {s.key: s for s in SETTINGS}
GROUPS: list[str] = list(dict.fromkeys(s.group for s in SETTINGS))


def load_config() -> dict:
    """Return the merged config (file overrides defaults). Never raises; on any
    problem it logs and falls back to defaults so the live runner is safe."""
    cfg = dict(DEFAULTS)
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            for k in DEFAULTS:
                if k in data:  # explicit file value wins (incl. None for optfloat = "off")
                    cfg[k] = data[k]
            log.info(f"Loaded ORB config overrides from {CONFIG_PATH.name}")
    except Exception as e:
        log.warning(f"Could not read {CONFIG_PATH.name} ({e}); using built-in defaults.")
    return cfg


def save_config(values: dict) -> Path:
    """Write only known keys to orb_config.json (pretty-printed). Returns the path."""
    out = {k: values[k] for k in DEFAULTS if k in values}
    CONFIG_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return CONFIG_PATH


def parse_time(s) -> Optional[time]:
    """'11:30' -> time(11,30); blank/None -> None."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def build_params(cfg: dict):
    """Construct strategies.orb.Params from a config dict (live-relevant fields)."""
    from strategies.orb import Params  # local import to avoid cycles
    return Params(
        or_minutes=int(cfg["or_minutes"]),
        target_r=float(cfg["target_r"]),
        risk_per_trade=float(cfg["risk_per_trade"]),
        max_position_pct=float(cfg["max_position_pct"]),
        max_position_dollars=(None if cfg["max_position_dollars"] in (None, "")
                              else float(cfg["max_position_dollars"])),
        move_stop_to_be_at_r=(None if cfg["move_stop_to_be_at_r"] in (None, "")
                              else float(cfg["move_stop_to_be_at_r"])),
        no_entry_after_time=parse_time(cfg["no_entry_after_time"]),
        stop_offset_pct=float(cfg["stop_offset_pct"]),
        enable_long=True,
        enable_short=bool(cfg["short_enabled"]),
        max_flips=int(cfg["max_flips"]),
    )
