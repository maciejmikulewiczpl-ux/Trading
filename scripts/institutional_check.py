"""Institutional-ownership check (the post's "step 1 - look who's buying").

For a set of tickers, pull from yfinance (free, 13F-derived):
  - institutionsPercentHeld : what fraction of shares institutions hold
  - top reported holders + each holder's pctChange since the prior filing
and summarize NET institutional accumulation: among the top reported holders,
how many ADDED vs TRIMMED and the net share delta. Positive net = funds were
net buyers as of the last 13F.

IMPORTANT LAG: 13F filings are due ~45 days after quarter-end, so this data is
typically 1.5-3 MONTHS old (e.g. mid-June shows the Mar-31 quarter). It confirms
a long-running accumulation story; it does NOT time anything. Treat as context,
exactly the caveat on the original method.

MUST run under .venv-openbb (yfinance lives there only):
    .venv-openbb/Scripts/python.exe scripts/institutional_check.py            # latest swing_screen CSV
    .venv-openbb/Scripts/python.exe scripts/institutional_check.py NUE,CAT,TSM
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def latest_screen_symbols() -> list[str]:
    """PASSing symbols from the most recent scripts/swing_screen_*.csv."""
    files = sorted(ROOT.glob("scripts/swing_screen_*.csv"))
    if not files:
        return []
    df = pd.read_csv(files[-1])
    if "PASS" in df.columns:
        df = df[df["PASS"] == True]  # noqa: E712
    print(f"(using {files[-1].name}: {len(df)} passing names)\n")
    return df["symbol"].tolist()


def check(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for i, sym in enumerate(symbols, 1):
        print(f"  {i}/{len(symbols)} {sym} ...", flush=True)
        inst_pct = n_inst = None
        as_of = None
        n_add = n_trim = 0
        net_shares = 0.0
        try:
            t = yf.Ticker(sym)
            mh = t.get_major_holders()
            if mh is not None and "Value" in mh.columns:
                v = mh["Value"]
                inst_pct = float(v.get("institutionsPercentHeld")) if "institutionsPercentHeld" in v.index else None
                n_inst = int(v.get("institutionsCount")) if "institutionsCount" in v.index else None
            ih = t.get_institutional_holders()
            if ih is not None and not ih.empty:
                as_of = str(ih["Date Reported"].iloc[0])[:10]
                for _, h in ih.iterrows():
                    pc = h.get("pctChange")
                    sh = h.get("Shares")
                    if pd.isna(pc) or pd.isna(sh):
                        continue
                    pc, sh = float(pc), float(sh)
                    if pc > 0:
                        n_add += 1
                    elif pc < 0:
                        n_trim += 1
                    # prior shares = sh / (1+pc); net delta = sh - prior
                    if pc > -1:
                        net_shares += sh - sh / (1.0 + pc)
        except Exception as e:
            print(f"     ({sym} failed: {str(e)[:70]})")
        rows.append({
            "symbol": sym,
            "inst_own%": round(inst_pct * 100, 1) if inst_pct is not None else None,
            "n_funds": n_inst,
            "top_adding": n_add, "top_trimming": n_trim,
            "net_shares_added": int(net_shares),
            "verdict": ("ACCUMULATING" if net_shares > 0 and n_add >= n_trim
                        else "DISTRIBUTING" if net_shares < 0 and n_trim > n_add
                        else "mixed"),
            "as_of": as_of,
        })
        time.sleep(0.4)   # polite to yahoo
    return pd.DataFrame(rows)


def main(argv) -> int:
    if len(argv) > 1 and argv[1].strip():
        syms = [s.strip().upper() for s in argv[1].split(",") if s.strip()]
    else:
        syms = latest_screen_symbols()
    if not syms:
        print("No symbols (pass a comma list, or run the screener first).")
        return 1

    df = check(syms)
    # rank by net accumulation
    df = df.sort_values("net_shares_added", ascending=False).reset_index(drop=True)
    pd.set_option("display.max_rows", None, "display.width", 160)
    print()
    print(df.to_string(index=False))
    out = ROOT / "scripts" / "institutional_check.csv"
    df.to_csv(out, index=False)
    print(f"\nLag note: data is as of the last 13F (~1.5-3 months old). "
          f"'ACCUMULATING' = funds were net buyers into that quarter-end. -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
