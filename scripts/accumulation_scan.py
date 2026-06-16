"""Quiet-accumulation scan: where did institutions BUY heavily into the last 13F, yet the
price has barely moved SINCE that quarter-end? Thesis: smart-money accumulation the market
hasn't repriced yet (the post's "is a big fund adding?" — but filtered to names that haven't
already run).

HONEST CAVEATS (told to the user): 13F data lags ~1.5-3 months (funds may have trimmed
since; "added" = change vs the PRIOR filing, i.e. last quarter). "Bought + flat = will move"
is a THESIS, not a validated edge. Research watchlist only — not a signal, not wired to money.

Scans UNIVERSE + EXPANSION (~257 liquid names). MUST run under .venv-openbb (yfinance):
    .venv-openbb/Scripts/python.exe scripts/accumulation_scan.py
    .venv-openbb/Scripts/python.exe scripts/accumulation_scan.py --max-move 0.05 --top 25
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from institutional_check import check  # noqa: E402  (reuse the 13F accumulation logic; yfinance-only)


def _list_literal(pyfile: Path, varname: str) -> list[str]:
    """Extract a list-of-strings literal from a .py file WITHOUT importing it (the source
    modules import alpaca, which isn't installed in .venv-openbb where yfinance lives)."""
    import ast
    tree = ast.parse(pyfile.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == varname for t in node.targets):
            return [s for s in ast.literal_eval(node.value) if isinstance(s, str)]
    return []


UNIVERSE = _list_literal(ROOT / "backtest" / "universe_scan.py", "UNIVERSE")
EXPANSION = _list_literal(ROOT / "backtest" / "fetch_universe_expanded.py", "EXPANSION")

MAX_MOVE = 0.07   # "didn't move much" = within +/-7% since the 13F as-of date


def price_since(symbols: list[str], as_of_by_sym: dict) -> dict:
    """{sym: (last_price, return_since_its_13f_as_of)} via one batch yfinance download."""
    import yfinance as yf
    df = yf.download(symbols, period="7mo", auto_adjust=True, progress=False)
    close = df["Close"] if "Close" in df else df
    out = {}
    for s in symbols:
        try:
            ser = (close[s] if isinstance(close, pd.DataFrame) else close).dropna()
            if ser.empty:
                out[s] = (None, None); continue
            last = float(ser.iloc[-1])
            asof = as_of_by_sym.get(s)
            base = None
            if asof:
                after = ser[ser.index >= pd.Timestamp(asof)]
                base = float(after.iloc[0]) if len(after) else None
            if base is None:
                base = float(ser.iloc[0])
            out[s] = (last, last / base - 1.0)
        except Exception:
            out[s] = (None, None)
    return out


def main(argv) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-move", type=float, default=MAX_MOVE, dest="max_move",
                    help="max |price move| since the 13F as-of date (0.07 = +/-7%%)")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv[1:])

    universe = sorted(set(UNIVERSE) | set(EXPANSION))
    print(f"Quiet-accumulation scan over {len(universe)} names "
          f"(institutions ADDING into last 13F + price within +/-{args.max_move:.0%} since)\n")

    df = check(universe, drop_artifacts=True)             # 13F accumulation, artifacts dropped
    as_of = dict(zip(df["symbol"], df["as_of"]))
    print("\nfetching price-since-13F (batch) ...", flush=True)
    ps = price_since(universe, as_of)
    df["price"] = df["symbol"].map(lambda s: ps.get(s, (None, None))[0])
    df["ret_since_13f"] = df["symbol"].map(lambda s: ps.get(s, (None, None))[1])

    # quiet accumulation = REAL net institutional buying (artifacts already dropped in check;
    # also exclude >100%-ownership data-artifact names) AND ~flat since the 13F
    df["quiet_accum"] = (
        (df["verdict"] == "ACCUMULATING")
        & (df["top_adding"] > df["top_trimming"])
        & (df["net_shares_added"] > 0)
        & (~df["own_artifact"])
        & (df["ret_since_13f"].notna())
        & (df["ret_since_13f"].abs() <= args.max_move)
    )
    # rank survivors: most net buying breadth, then magnitude
    df["score"] = (df["top_adding"] - df["top_trimming"]) + df["net_shares_added"] / 1e8
    out = df.sort_values(["quiet_accum", "score"], ascending=[False, False]).reset_index(drop=True)

    csv = ROOT / "scripts" / "accumulation_scan.csv"
    out.to_csv(csv, index=False)
    hits = out[out["quiet_accum"]]
    cols = ["symbol", "price", "ret_since_13f", "inst_own%", "top_adding", "top_trimming",
            "net_shares_added", "n_artifact_holders", "as_of"]
    pd.set_option("display.max_rows", None, "display.width", 170)
    show = hits.head(args.top).copy()
    if not show.empty:
        show["ret_since_13f"] = (show["ret_since_13f"] * 100).round(1)
    print(f"\n=== {len(hits)} names: institutions ACCUMULATING + flat (<= {args.max_move:.0%}) since 13F ===")
    print(show[cols].to_string(index=False) if not show.empty else "  (none matched)")
    print(f"\nFull table -> {csv.name}.  ret_since_13f = % move since the holding's report date.")
    print("CAVEAT: 13F is ~1.5-3mo stale; this is a research watchlist, not a signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
