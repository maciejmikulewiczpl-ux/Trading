"""Reconnaissance: which OpenBB earnings endpoints work without paid API keys?

Strategy: brute-force try every endpoint x candidate-free-provider combo, log
what succeeds. Print shape and sample for working ones.

Run with:
    .\\.venv-openbb\\Scripts\\python.exe scripts\\earnings_recon.py
"""
from __future__ import annotations

import io
import sys
from datetime import date, timedelta

# Force UTF-8 stdout on Windows so emoji/symbols don't crash.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from openbb import obb


FREE_PROVIDERS = ["yfinance", "nasdaq", "sec", "seeking_alpha", "tmx"]
# Endpoints relevant to PEAD, paired with method-path and probe kwargs.
PROBES = {
    "equity.calendar.earnings": ("calendar.earnings", {
        "start_date": date.today() - timedelta(days=14),
        "end_date":   date.today() + timedelta(days=14),
    }),
    "equity.fundamental.historical_eps": ("fundamental.historical_eps", {"symbol": "AAPL"}),
    "equity.fundamental.income":         ("fundamental.income",         {"symbol": "AAPL", "limit": 8}),
    "equity.estimates.historical":       ("estimates.historical",       {"symbol": "AAPL"}),
    "equity.estimates.consensus":        ("estimates.consensus",        {"symbol": "AAPL"}),
    "equity.estimates.forward_eps":      ("estimates.forward_eps",      {"symbol": "AAPL"}),
}


def call_endpoint(path: str, provider: str, base_kwargs: dict):
    fn = obb.equity
    for part in path.split("."):
        fn = getattr(fn, part)
    kwargs = {**base_kwargs, "provider": provider}
    return fn(**kwargs)


def main():
    print(f"OpenBB Platform version: {obb.system.version}")
    print()
    print(f"Trying providers: {FREE_PROVIDERS}")
    print()

    results = {}
    for endpoint, (path, base_kwargs) in PROBES.items():
        print("-" * 72)
        print(f"{endpoint}   (kwargs: {dict((k, str(v)) for k,v in base_kwargs.items())})")
        print("-" * 72)
        for prov in FREE_PROVIDERS:
            try:
                r = call_endpoint(path, prov, base_kwargs)
                df = r.to_df()
                if df.empty:
                    print(f"  [{prov:14s}] OK but EMPTY (0 rows)")
                    continue
                cols = list(df.columns)
                idx_name = df.index.name
                idx_range = ""
                try:
                    idx_range = f"  index range: {df.index.min()} -> {df.index.max()}"
                except Exception:
                    pass
                print(f"  [{prov:14s}] OK  rows={len(df)}  cols={len(cols)}{idx_range}")
                results.setdefault(endpoint, []).append((prov, df))
            except Exception as e:
                msg = str(e).split("\n")
                first = next((ln.strip() for ln in msg if ln.strip()), "")
                if len(first) > 110:
                    first = first[:107] + "..."
                print(f"  [{prov:14s}] FAIL  {first}")
        print()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for endpoint in PROBES:
        good = [p for p, _ in results.get(endpoint, [])]
        if good:
            print(f"  [OK]   {endpoint}  -> {good}")
        else:
            print(f"  [FAIL] {endpoint}")

    if not results:
        print()
        print("No endpoint returned data from any free provider.")
        return

    print()
    print("=" * 72)
    print("SAMPLE DATA from each working endpoint+provider")
    print("=" * 72)
    for endpoint, hits in results.items():
        for prov, df in hits:
            print()
            print("-" * 72)
            print(f"{endpoint}   via   {prov}")
            print("-" * 72)
            print(f"columns: {list(df.columns)}")
            try:
                print(f"index:   name={df.index.name}  type={type(df.index).__name__}")
            except Exception:
                pass
            print()
            # Try to print first 8 rows compactly
            with_head = df.head(8)
            print(with_head.to_string()[:2500])


if __name__ == "__main__":
    main()
