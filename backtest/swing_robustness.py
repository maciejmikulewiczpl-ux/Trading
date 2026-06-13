"""swing_robustness.py -- plateau sweep for the swing-engine candidate variant.

Grid: entry_days {20, 55} x stop_mult {2.0, 2.5, 3.0} x compression {on, off}
= 12 cells. PASS requires >= 2/3 of cells (>=8) within 30% of the candidate's
Sharpe. A knife-edge = overfit = FAIL.

Run ONLY after compare_swing_variants.py confirms >= 1 gate pass:
    .venv-openbb\\Scripts\\python.exe backtest\\swing_robustness.py --variant V1
"""
from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backtest.run_swing import run_simulation, CACHE  # noqa: E402
from backtest.compare_swing_variants import sharpe  # noqa: E402

ENTRY_DAYS = [20, 55]
STOP_MULTS = [2.0, 2.5, 3.0]
COMPRESS = [True, False]


def main() -> None:
    variant = "V1"
    for i, arg in enumerate(sys.argv):
        if arg == "--variant" and i + 1 < len(sys.argv):
            variant = sys.argv[i + 1]

    print(f"=== swing_robustness.py  candidate: {variant} ===")
    print("Loading cache...")
    data = pickle.load(open(CACHE, "rb"))

    # canonical params for the candidate
    canon_entry = 20 if variant == "V2" else 55
    canon_stop = 2.5
    canon_comp = (variant == "V1")

    results = []
    total = len(ENTRY_DAYS) * len(STOP_MULTS) * len(COMPRESS)
    done = 0
    for ed in ENTRY_DAYS:
        for sm in STOP_MULTS:
            for cp in COMPRESS:
                label = f"entry={ed}d  stop={sm}x  comp={'on' if cp else 'off'}"
                is_canon = (ed == canon_entry and sm == canon_stop and cp == canon_comp)
                trades, daily = run_simulation(
                    data, variant="V0",
                    entry_days=ed, stop_mult=sm, use_compression=cp,
                )
                sh = sharpe(daily)
                pnl = sum(t.pnl_net for t in trades)
                results.append((label, sh, pnl, is_canon))
                done += 1
                flag = " <-- CANDIDATE" if is_canon else ""
                print(f"  [{done:>2}/{total}] {label:<40} Sharpe={sh:>5.2f}  "
                      f"PnL=${pnl:>+9,.0f}{flag}")

    # find candidate Sharpe
    canon_sharpe = next(sh for _, sh, _, ic in results if ic)

    # plateau check: >= 2/3 cells within 30% of candidate Sharpe
    within_30 = sum(1 for _, sh, _, _ in results if abs(sh - canon_sharpe) / max(abs(canon_sharpe), 0.01) <= 0.30)
    threshold = math.ceil(2 * len(results) / 3)
    plateau_pass = within_30 >= threshold

    print(f"\n{'='*70}")
    print(f"  Candidate Sharpe: {canon_sharpe:.2f}")
    print(f"  Cells within 30% of candidate: {within_30}/{len(results)} "
          f"(need >= {threshold} for PASS)")
    print(f"  Plateau verdict: {'PASS' if plateau_pass else 'FAIL (knife-edge -- overfit)'}")
    if plateau_pass:
        print(f"  -> Robustness confirmed. Ship path: paper runner on VM.")
    else:
        print(f"  -> Knife-edge: results depend on exact parameters. Do not ship.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
