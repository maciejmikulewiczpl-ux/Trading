"""PATTERN HUNT for news-edge — is there an edge we're missing, or is it dead?

The built-in analyze shows claude(+) is flat vs a gapper screen. Here we dig where means wash
out: characterize the WINNERS among the tradeable set (signal=1, long-only, what the bot buys),
by THEME and by feature COMBINATIONS never examined, then test whether any refined subset beats
the mechanical >3%-gapper CONTROL out-of-sample. Metric = ret_945_close (9:45->close, the
same-day horizon the bot measures). Honest: small n per cell -> OOS-split + mechanism required.
Run: .venv/Scripts/python.exe experiments/news_edge/pattern_hunt.py"""
from __future__ import annotations
import glob, json, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PICKS = ROOT / "experiments" / "news_edge" / "picks"
WIN = 5.0     # ret_945_close >= +5% == a meaningful same-day catalyst pop


def load():
    rows, ctrl = [], []
    for f in sorted(glob.glob(str(PICKS / "2026-*.json"))):
        rec = json.load(open(f)); ps = rec if isinstance(rec, list) else rec.get("picks", rec)
        date = Path(f).stem
        for p in ps:
            if not isinstance(p, dict) or p.get("ret_945_close") is None:
                continue
            r = {"date": date, "sym": p.get("symbol"), "r": float(p["ret_945_close"]),
                 "sig": p.get("signal"), "conf": p.get("confidence"), "gap": p.get("gap_pct"),
                 "rvol": p.get("premarket_rvol"), "earn": p.get("earnings_day"),
                 "theme": p.get("theme"), "srcsig": p.get("source_signals") or {}}
            if p.get("control"):
                ctrl.append(r)
            elif p.get("signal") in (1, "1"):     # the bot trades signal=1 (long-only)
                rows.append(r)
    return rows, ctrl


def _stat(rs):
    if not rs:
        return None
    v = [x["r"] for x in rs]
    return len(v), statistics.mean(v), sum(1 for x in v if x > 0) / len(v) * 100, \
        sum(1 for x in v if x >= WIN) / len(v) * 100, max(v)


def main():
    rows, ctrl = load()
    cs = _stat(ctrl); bs = _stat(rows)
    print("=" * 78)
    print(f"NEWS PATTERN HUNT | signal=1 picks: {len(rows)} | control(>3% gappers): {len(ctrl)} | metric ret_945")
    print(f"BASELINES: signal=1 mean {bs[1]:+.2f}% win {bs[2]:.0f}% (win>=+5% {bs[3]:.0f}%) | "
          f"control mean {cs[1]:+.2f}%  -> analyst edge vs control {bs[1]-cs[1]:+.2f}pp")
    print("=" * 78)

    # 1. THEME -- never examined
    print("\n=== BY THEME (signal=1) -- n>=6 shown, sorted by mean ret_945 ===")
    byt = defaultdict(list)
    for r in rows:
        byt[r["theme"] or "none"].append(r)
    print(f"  {'theme':<20}{'n':>4}{'mean%':>8}{'win%':>7}{'>=+5%':>7}{'best%':>8}")
    for th, rs in sorted(byt.items(), key=lambda x: -(_stat(x[1])[1])):
        if len(rs) < 6:
            continue
        n, m, w, w5, bst = _stat(rs)
        flag = "  <-- beats control" if m > cs[1] + 0.5 else ""
        print(f"  {th:<20}{n:>4}{m:>+8.2f}{w:>7.0f}{w5:>7.0f}{bst:>+8.1f}{flag}")

    # 2. FEATURE COMBOS -- stack the known-good splits, test vs control
    def sub(pred):
        return [r for r in rows if pred(r)]
    big = lambda r: (r["gap"] is not None and abs(r["gap"]) >= 3)
    noearn = lambda r: not r["earn"]
    locon = lambda r: (r["conf"] is not None and r["conf"] < 0.6)
    combos = {
        "non-earnings": noearn,
        "big-gap (|gap|>=3)": big,
        "low-conf (<0.6)": locon,
        "non-earn + big-gap": lambda r: noearn(r) and big(r),
        "non-earn + low-conf": lambda r: noearn(r) and locon(r),
        "big-gap + low-conf": lambda r: big(r) and locon(r),
        "non-earn+big-gap+low-conf": lambda r: noearn(r) and big(r) and locon(r),
    }
    print("\n=== FEATURE COMBOS (signal=1) vs control -- does stacking build a real edge? ===")
    print(f"  {'subset':<30}{'n':>4}{'mean%':>8}{'win%':>7}{'vs ctrl':>9}")
    for name, pred in combos.items():
        st = _stat(sub(pred))
        if st and st[0] >= 6:
            print(f"  {name:<30}{st[0]:>4}{st[1]:>+8.2f}{st[2]:>7.0f}{st[1]-cs[1]:>+9.2f}")

    # 3. OOS split on the best-looking combo
    dates = sorted({r["date"] for r in rows}); mid = dates[len(dates) // 2]
    best = lambda r: noearn(r) and big(r)
    print(f"\n=== OOS split on 'non-earn + big-gap' (early<{mid}<=late) ===")
    for lbl, sel in [("EARLY", [r for r in rows if r["date"] < mid]), ("LATE", [r for r in rows if r["date"] >= mid])]:
        s = _stat([r for r in sel if best(r)]); c = _stat([r for r in ctrl if r["date"] < mid] if lbl == "EARLY" else [r for r in ctrl if r["date"] >= mid])
        if s and c:
            print(f"  {lbl}: subset mean {s[1]:+.2f}% (n={s[0]}) vs control {c[1]:+.2f}%  -> edge {s[1]-c[1]:+.2f}pp")

    # 4. source agreement
    print("\n=== SOURCE AGREEMENT (# of sources firing on the pick) ===")
    byk = defaultdict(list)
    for r in rows:
        byk[len([v for v in r["srcsig"].values() if v])].append(r)
    for k in sorted(byk):
        st = _stat(byk[k])
        if st and st[0] >= 5:
            print(f"  {k} source(s): n={st[0]:>3}  mean {st[1]:+.2f}%  win {st[2]:.0f}%")

    # 5. the actual winners
    print(f"\n=== BIG WINNERS (signal=1, ret_945 top 10) ===")
    for r in sorted(rows, key=lambda x: -x["r"])[:10]:
        print(f"  {r['date']} {r['sym']:6} {r['r']:>+7.1f}%  theme={r['theme']} gap={r['gap']} "
              f"conf={r['conf']} earn={r['earn']}")
    print("\n[caveats] small per-cell n -> multiple-comparison risk; a subset counts only if it beats "
          "control, survives the OOS split, and has a mechanism. Hypotheses, not a ship signal.")


if __name__ == "__main__":
    main()
