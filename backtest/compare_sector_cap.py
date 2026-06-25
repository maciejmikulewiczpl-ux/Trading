"""Does a PER-SECTOR cap on concurrent positions improve the diversified book?

The diversification win (compare_diversification.py: risk across ~16 names ~2x Sharpe,
~half drawdown) quietly assumes the 16 positions are INDEPENDENT. But the cap is
first-come, so on a chip-news morning 16 semis can fill the whole book -> a concentrated
sector bet wearing a diversification costume; they stop out together (the tail the
diversification was meant to remove). This caps concurrent positions PER correlation-
cluster sector (max N per sector) on top of the total cap, and asks whether forcing real
breadth lifts Sharpe / cuts drawdown vs the current sector-blind cap.

Exit/sizing held CONSTANT across variants (fixed-2R, cap 16, half-risk vol dial) so the
only thing that changes is the sector cap -> a clean read of the cap's effect. 730d, OOS
halved. Gate: a sector cap earns its keep only if Sharpe >= baseline AND maxDD better,
in BOTH halves (PnL may dip slightly — that's an acceptable price for real drawdown cuts).

Run:  .venv/Scripts/python.exe backtest/compare_sector_cap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.compare_exits import load  # noqa: E402
from backtest.compare_norefill_trend import trend_eligibility, apply_filter  # noqa: E402
from backtest.compare_volpause import prior_vol_flags, series, perf, CAP  # noqa: E402
from backtest.compare_selection import _tday  # noqa: E402

WINDOW = "730"
SECTOR_CAPS = [None, 6, 5, 4, 3]   # None = current (sector-blind) baseline

# Correlation-cluster sectors (coarser than GICS where co-movement is what matters:
# semis as their own cluster, mega-platforms together, etc.). Unmapped -> 'OTHER'.
SECTORS = {
    # broad ETFs (move together with the index)
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    # semiconductors (tightest co-movement)
    "NVDA": "SEMI", "AVGO": "SEMI", "AMD": "SEMI", "INTC": "SEMI", "QCOM": "SEMI",
    "TXN": "SEMI", "MU": "SEMI", "AMAT": "SEMI", "LRCX": "SEMI", "ARM": "SEMI",
    "MRVL": "SEMI", "ADI": "SEMI", "KLAC": "SEMI", "SNPS": "SEMI", "CDNS": "SEMI",
    "MCHP": "SEMI", "ON": "SEMI", "MPWR": "SEMI", "TSM": "SEMI", "ASML": "SEMI", "SMCI": "SEMI",
    # mega-cap platforms
    "AAPL": "MEGATECH", "MSFT": "MEGATECH", "GOOGL": "MEGATECH", "AMZN": "MEGATECH", "META": "MEGATECH",
    # software / IT services
    "ORCL": "SW", "ADBE": "SW", "CRM": "SW", "NOW": "SW", "PANW": "SW", "CRWD": "SW",
    "INTU": "SW", "SHOP": "SW", "IBM": "SW", "ACN": "SW", "PLTR": "SW", "FTNT": "SW",
    "DDOG": "SW", "NET": "SW", "SNOW": "SW", "ZS": "SW", "TEAM": "SW", "WDAY": "SW",
    # internet / media / streaming
    "NFLX": "INTERNET", "ABNB": "INTERNET", "UBER": "INTERNET", "BABA": "INTERNET",
    "DIS": "MEDIA", "CMCSA": "MEDIA", "SPOT": "MEDIA", "RBLX": "INTERNET", "PINS": "INTERNET",
    "SNAP": "INTERNET", "DASH": "INTERNET", "ROKU": "INTERNET", "TTD": "INTERNET",
    # payments / fintech
    "V": "FIN", "MA": "FIN", "PYPL": "FIN", "AXP": "FIN", "SOFI": "FIN", "HOOD": "FIN", "COIN": "CRYPTO", "MARA": "CRYPTO",
    # banks / capital markets
    "JPM": "FIN", "BAC": "FIN", "WFC": "FIN", "GS": "FIN", "MS": "FIN", "C": "FIN",
    "USB": "FIN", "SCHW": "FIN", "BLK": "FIN", "SPGI": "FIN", "ICE": "FIN", "CME": "FIN",
    "PNC": "FIN", "TFC": "FIN", "COF": "FIN", "BK": "FIN",
    # healthcare / pharma
    "UNH": "HEALTH", "JNJ": "HEALTH", "LLY": "HEALTH", "PFE": "HEALTH", "MRK": "HEALTH",
    "ABBV": "HEALTH", "TMO": "HEALTH", "ABT": "HEALTH", "DHR": "HEALTH", "AMGN": "HEALTH",
    "ISRG": "HEALTH", "BMY": "HEALTH", "ELV": "HEALTH", "CVS": "HEALTH", "GILD": "HEALTH",
    "VRTX": "HEALTH", "REGN": "HEALTH", "MRNA": "HEALTH", "SYK": "HEALTH", "MDT": "HEALTH",
    # consumer staples
    "WMT": "STAPLES", "COST": "STAPLES", "KO": "STAPLES", "PEP": "STAPLES", "PG": "STAPLES",
    "CL": "STAPLES", "MDLZ": "STAPLES", "MO": "STAPLES", "PM": "STAPLES", "MNST": "STAPLES",
    "KDP": "STAPLES", "STZ": "STAPLES", "GIS": "STAPLES",
    # consumer discretionary / retail / restaurants
    "HD": "DISC", "NKE": "DISC", "SBUX": "DISC", "MCD": "DISC", "TGT": "DISC", "LOW": "DISC",
    "BKNG": "DISC", "LULU": "DISC", "ROST": "DISC", "CMG": "DISC", "TJX": "DISC", "MAR": "DISC",
    # autos
    "TSLA": "AUTO", "F": "AUTO", "GM": "AUTO",
    # energy
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY", "SLB": "ENERGY", "EOG": "ENERGY",
    "MPC": "ENERGY", "PSX": "ENERGY", "VLO": "ENERGY", "OXY": "ENERGY", "KMI": "ENERGY",
    "OKE": "ENERGY", "WMB": "ENERGY",
    # industrials
    "CAT": "INDU", "BA": "INDU", "GE": "INDU", "HON": "INDU", "RTX": "INDU", "LMT": "INDU",
    "DE": "INDU", "UNP": "INDU", "ETN": "INDU", "MMM": "INDU", "UPS": "INDU", "FDX": "INDU",
    # telecom / utility / materials / RE
    "T": "TELCO", "VZ": "TELCO", "TMUS": "TELCO",
    "NEE": "UTIL", "DUK": "UTIL", "SO": "UTIL",
    "LIN": "MATERIALS", "SHW": "MATERIALS", "FCX": "MATERIALS", "NEM": "MATERIALS", "NUE": "MATERIALS",
    "AMT": "REIT", "PLD": "REIT", "EQIX": "REIT", "PSA": "REIT", "O": "REIT", "CCI": "REIT",
}


def sector_portfolio(trades, cap, sector_cap):
    """Greedy-by-time fill: max `cap` concurrent total AND max `sector_cap` per sector."""
    taken, open_total = [], []
    open_by_sec: dict[str, list] = {}
    for t in sorted(trades, key=lambda x: x.entry_time):
        open_total = [x for x in open_total if x > t.entry_time]
        sec = SECTORS.get(t.symbol, "OTHER")
        cur = [x for x in open_by_sec.get(sec, []) if x > t.entry_time]
        open_by_sec[sec] = cur
        if (cap is None or len(open_total) < cap) and (sector_cap is None or len(cur) < sector_cap):
            taken.append(t)
            open_total.append(t.exit_time)
            cur.append(t.exit_time)
    return taken


def main():
    all_bars, days, present, cached15, closes = load(int(WINDOW))
    elig = trend_eligibility(closes, present, days)
    hv = prior_vol_flags(closes, days)
    mid = sorted(days)[len(days) // 2]
    mult = {d: (0.5 if hv[d] else 1.0) for d in days}
    d1 = [d for d in days if d < mid]
    d2 = [d for d in days if d >= mid]
    # live edge: tight-OR (<=0.5% of price) + trend filter (fixed-2R exit is the caveat)
    tight = [t for t in cached15 if abs(t.entry_price - t.stop_price) / t.entry_price <= 0.005]
    filtered = apply_filter(tight, elig)

    # coverage check: how many traded symbols are unmapped?
    syms = {t.symbol for t in filtered}
    unmapped = sorted(s for s in syms if s not in SECTORS)
    print(f"sector map covers {len(syms) - len(unmapped)}/{len(syms)} traded symbols; "
          f"unmapped->OTHER: {unmapped[:20]}")

    print(f"\n=== {WINDOW}d per-SECTOR cap (total cap {CAP}, fixed-2R, vol dial), OOS {mid} ===")
    print(f"{'per-sector cap':<16}{'trades':>8}{'PnL$':>11}{'Sharpe':>8}{'maxDD$':>10}{'h1 Sh':>7}{'h2 Sh':>7}")
    print("-" * 67)
    for sc in SECTOR_CAPS:
        taken = sector_portfolio(filtered, CAP, sc)
        s = perf(series(taken, days, mult))
        s1 = perf(series([t for t in taken if _tday(t) < mid], d1, mult))
        s2 = perf(series([t for t in taken if _tday(t) >= mid], d2, mult))
        lbl = "none (baseline)" if sc is None else f"max {sc}/sector"
        print(f"{lbl:<16}{len(taken):>8}{s['pnl']:>+11,.0f}{s['sharpe']:>8.2f}"
              f"{s['maxdd']:>10,.0f}{s1['sharpe']:>7.2f}{s2['sharpe']:>7.2f}")
    print("\nGate: a sector cap earns it only if Sharpe >= baseline AND maxDD better in BOTH")
    print("halves. If caps just cut trades for the same/worse Sharpe, the book wasn't actually")
    print("over-concentrated -> reject. Exit held constant (fixed-2R) -> relative read only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
