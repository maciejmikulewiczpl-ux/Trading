"""Verify sync_existing_orders_today + prebuild_or_if_late populate state
correctly against today's actual Alpaca orders and bars. Read-only.

Run:
    .venv/Scripts/python.exe scripts/smoke_rehydration.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from live.paper_orb import (
    ET,
    PARAMS,
    RTH_OPEN,
    RunState,
    SymbolState,
    WATCHLIST_DEFAULT,
    build_clients,
    combine_et,
    detect_untracked_positions,
    load_env,
    prebuild_or_if_late,
    sync_existing_orders_today,
)


def main() -> int:
    load_env()
    tc, dc = build_clients()
    watchlist = WATCHLIST_DEFAULT
    today = datetime.now(ET).date()

    run = RunState(states={s: SymbolState() for s in watchlist})

    print(f"Rehydrating for {today} (watchlist: {', '.join(watchlist)})")
    print("=" * 70)

    sync_existing_orders_today(tc, run, today)
    detect_untracked_positions(tc, run, watchlist)

    open_dt = combine_et(today, RTH_OPEN)
    or_end_dt = open_dt + timedelta(minutes=PARAMS.or_minutes)
    prebuild_or_if_late(dc, run, watchlist, today, or_end_dt)

    print()
    print(f"{'Sym':<6}{'entered':<9}{'exited':<8}{'or_lock':<9}"
          f"{'entry':>10}{'exit':>10}{'stop':>10}{'target':>10}{'PnL':>10}")
    print("-" * 86)
    for sym in watchlist:
        st = run.states[sym]
        entry = f"${st.entry_price:.2f}" if st.entry_price is not None else "-"
        exit_p = f"${st.exit_price:.2f}" if st.exit_price is not None else "-"
        stop = f"${st.stop_price:.2f}" if st.stop_price is not None else "-"
        target = f"${st.target_price:.2f}" if st.target_price is not None else "-"
        pnl = f"${st.realized_pnl:+,.0f}" if st.realized_pnl is not None else "-"
        print(f"{sym:<6}{str(st.entered):<9}{str(st.exited):<8}"
              f"{str(st.or_locked):<9}{entry:>10}{exit_p:>10}{stop:>10}"
              f"{target:>10}{pnl:>10}")
        if st.or_locked and st.or_high is not None and st.or_low is not None:
            print(f"      OR: high=${st.or_high:.2f}  low=${st.or_low:.2f}")
        if st.exited:
            print(f"      EXIT REASON: {st.exit_reason}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
