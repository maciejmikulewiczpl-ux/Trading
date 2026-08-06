"""PATTERN HUNT — reverse-engineer the WINNERS instead of testing signals one-at-a-time as a mean.

The strategy is tail-driven (money = the rare +50/+100/+1000% name), so mean-based signal tests
(multiday_selector.py) mostly wash out. Here we flip it: take every BIG ret_3d winner across all
scored boards and ask what they shared at pick time -- signal levels, PRICE, and 2-signal combos --
vs the base population. Tail-rate framing (P(win) by feature), not mean. OOS split on the top find.

HONEST GUARDs: few big winners (small tail) -> multiple-comparison risk is high; a pattern only counts
if it (a) is monotone-ish, (b) survives an early/late split, and (c) has a plausible mechanism. This
GENERATES hypotheses to forward-test, it does not authorize a live change.
Run: .venv/Scripts/python.exe experiments/lottery/pattern_hunt.py"""
from __future__ import annotations
import glob, json, os, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PICKS = ROOT / "experiments" / "lottery" / "picks"
WIN = 20.0        # ret_3d >= +20% == the pre-registered W3 explosive winner (the tail we care about)

# signal -> higher_is_stronger (ranks are lower=stronger)
SIGS = {"combined_score": True, "opt_expmove": True, "ignition": True, "pm_rvol": True,
        "gap_pct": True, "gtrends_spike": True, "wsb_surge": True, "squeeze": True,
        "realized_vol": True, "finra_short_ratio": True, "st_rank": False, "wsb_rank": False}


def load():
    rows = []
    for f in sorted(glob.glob(str(PICKS / "2026-*.json"))):
        rec = json.load(open(f)); rec = rec if isinstance(rec, list) else rec.get("picks", rec)
        date = Path(f).stem
        for c in rec:
            if isinstance(c, dict) and c.get("ret_3d") is not None:
                s = dict(c.get("signals") or {}); s["combined_score"] = c.get("combined_score")
                rows.append({"date": date, "sym": c["symbol"], "r3": float(c["ret_3d"]), "sig": s})
    return rows


def add_prices(rows):
    """Reconstruct each pick-date OPEN price via one batched Alpaca daily-bars fetch (IEX)."""
    for env in (".env.lottery", ".env"):
        p = ROOT / env
        if p.exists():
            for l in p.read_text().splitlines():
                l = l.strip()
                if l and not l.startswith("#") and "=" in l:
                    k, v = l.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed
        from datetime import datetime, timezone
        dc = StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])
    except Exception as e:
        print("price fetch unavailable:", str(e)[:60]); return
    syms = sorted({r["sym"] for r in rows})
    px = {}
    for i in range(0, len(syms), 100):
        grp = syms[i:i + 100]
        try:
            df = dc.get_stock_bars(StockBarsRequest(symbol_or_symbols=grp, timeframe=TimeFrame.Day,
                 start=datetime(2026, 6, 10, tzinfo=timezone.utc), end=datetime(2026, 8, 2, tzinfo=timezone.utc),
                 feed=DataFeed.IEX)).df
            for s in grp:
                try:
                    for t, r in df.xs(s, level=0).iterrows():
                        px[(s, str(t.date()))] = float(r.open)
                except KeyError:
                    pass
        except Exception:
            pass
    for r in rows:
        r["price"] = px.get((r["sym"], r["date"]))


def _val(r, sig, higher):
    v = r["sig"].get(sig)
    if v is None:
        return None
    return v if higher else -v


def winner_profile(rows):
    wins = [r for r in rows if r["r3"] >= WIN]
    print(f"\n=== WINNER PROFILE ===  {len(wins)}/{len(rows)} candidate-days are W3 winners (ret_3d>=+{WIN:.0f}%)")
    print(f"  {'feature':<18}{'win median':>12}{'all median':>12}{'coverage':>10}")
    feats = list(SIGS) + ["price"]
    for feat in feats:
        if feat == "price":
            wv = [r.get("price") for r in wins if r.get("price")]
            av = [r.get("price") for r in rows if r.get("price")]
        else:
            wv = [r["sig"].get(feat) for r in wins if r["sig"].get(feat) is not None]
            av = [r["sig"].get(feat) for r in rows if r["sig"].get(feat) is not None]
        if len(wv) < 4:
            continue
        wm, am = statistics.median(wv), statistics.median(av)
        flag = "  <-- winners skew" if abs(wm - am) > 0.25 * (abs(am) + 1e-9) else ""
        print(f"  {feat:<18}{wm:>12.3f}{am:>12.3f}{len(wv)/max(len(wins),1)*100:>9.0f}%{flag}")


def tail_by_quintile(rows):
    print(f"\n=== TAIL RATE by feature quintile === P(ret_3d>=+{WIN:.0f}%) in top-20% vs bottom-20% by each feature")
    print(f"  {'feature':<18}{'n':>5}{'Q5 tail%':>10}{'Q1 tail%':>10}{'Q5 mean r3':>12}{'spread':>9}")
    feats = [(s, SIGS[s]) for s in SIGS] + [("price", False)]  # price: cheaper=better -> lower is stronger
    scored = []
    for feat, higher in feats:
        vals = [(r, _val(r, feat, higher) if feat != "price" else (-r["price"] if r.get("price") else None))
                for r in rows]
        vals = [(r, v) for r, v in vals if v is not None]
        if len(vals) < 25:
            continue
        vals.sort(key=lambda x: x[1])
        q = len(vals) // 5
        q1, q5 = vals[:q], vals[-q:]
        t1 = sum(1 for r, _ in q1 if r["r3"] >= WIN) / len(q1) * 100
        t5 = sum(1 for r, _ in q5 if r["r3"] >= WIN) / len(q5) * 100
        m5 = statistics.mean([r["r3"] for r, _ in q5])
        scored.append((t5 - t1, feat, len(vals), t5, t1, m5))
    for spread, feat, n, t5, t1, m5 in sorted(scored, key=lambda x: -x[0]):
        print(f"  {feat:<18}{n:>5}{t5:>9.1f}%{t1:>9.1f}%{m5:>12.2f}{spread:>+9.1f}")


def interactions(rows):
    """A few pre-specified 2-feature combos: does the intersection have a higher tail rate than base?"""
    base = sum(1 for r in rows if r["r3"] >= WIN) / len(rows) * 100
    def topq(feat, higher, frac=0.4):
        vals = [(r, _val(r, feat, higher) if feat != "price" else (-r["price"] if r.get("price") else None)) for r in rows]
        vals = [(r, v) for r, v in vals if v is not None]; vals.sort(key=lambda x: -x[1])
        return {id(r) for r, _ in vals[:int(len(vals) * frac)]}
    combos = [("opt_expmove", True, "price", False), ("realized_vol", True, "price", False),
              ("opt_expmove", True, "ignition", True), ("gap_pct", True, "opt_expmove", True),
              ("ignition", True, "wsb_surge", True), ("realized_vol", True, "opt_expmove", True)]
    print(f"\n=== 2-FEATURE INTERACTIONS === base P(win)={base:.1f}% | tail rate when BOTH in top-40%")
    print(f"  {'combo':<34}{'n':>5}{'tail%':>8}{'lift':>7}")
    out = []
    for a, ha, b, hb in combos:
        A, B = topq(a, ha), topq(b, hb); inter = [r for r in rows if id(r) in A and id(r) in B]
        if len(inter) < 8:
            continue
        tr = sum(1 for r in inter if r["r3"] >= WIN) / len(inter) * 100
        out.append((tr / base if base else 0, f"{a}+{b}", len(inter), tr))
    for lift, name, n, tr in sorted(out, key=lambda x: -x[0]):
        print(f"  {name:<34}{n:>5}{tr:>7.1f}%{lift:>6.1f}x")


def list_monsters(rows):
    print("\n=== THE ACTUAL BIG WINNERS (ret_3d, top 12) ===")
    for r in sorted(rows, key=lambda x: -x["r3"])[:12]:
        s = r["sig"]
        prof = f"px={r.get('price') and round(r['price'],1)} vol={s.get('realized_vol')} ign={s.get('ignition')} " \
               f"gap={s.get('gap_pct')} opt={s.get('opt_expmove')} cs={s.get('combined_score')}"
        print(f"  {r['date']} {r['sym']:6} ret_3d {r['r3']:>+8.1f}%  | {prof}")


def oos_check(rows, feat, higher):
    dates = sorted({r["date"] for r in rows}); mid = dates[len(dates) // 2]
    print(f"\n=== OOS split on '{feat}' (top-40% tail rate, early vs late) ===")
    for lbl, sub in [("EARLY", [r for r in rows if r["date"] < mid]), ("LATE", [r for r in rows if r["date"] >= mid])]:
        vals = [(r, _val(r, feat, higher) if feat != "price" else (-r["price"] if r.get("price") else None)) for r in sub]
        vals = [(r, v) for r, v in vals if v is not None]; vals.sort(key=lambda x: -x[1])
        top = vals[:int(len(vals) * 0.4)]
        if not top:
            continue
        tr = sum(1 for r, _ in top if r["r3"] >= WIN) / len(top) * 100
        base = sum(1 for r in sub if r["r3"] >= WIN) / len(sub) * 100
        print(f"  {lbl}: top-40% tail {tr:.1f}% vs base {base:.1f}%  (lift {tr/base if base else 0:.1f}x, n_top={len(top)})")


def main():
    rows = load()
    print(f"loaded {len(rows)} scored candidate-days; reconstructing prices...")
    add_prices(rows)
    npx = sum(1 for r in rows if r.get("price"))
    print(f"prices for {npx}/{len(rows)}")
    winner_profile(rows)
    tail_by_quintile(rows)
    interactions(rows)
    list_monsters(rows)
    oos_check(rows, "opt_expmove", True)
    oos_check(rows, "price", False)
    print("\n[caveats] small tail (few W3 winners) -> high multiple-comparison risk; treat any pattern as a "
          "HYPOTHESIS to forward-test, robust only if monotone + survives the OOS split + has a mechanism.")


if __name__ == "__main__":
    main()
