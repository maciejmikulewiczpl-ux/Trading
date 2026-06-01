"""One-shot check: DBC's trailing returns AS-OF the dual-momentum signal date
vs today, plus last 30 daily closes. Helps interpret "DBC has been weak lately"
in context of the system's 3/6/12-month blended signal."""
import sys
import pandas as pd
import yfinance as yf

dbc = yf.download("DBC", start="2024-12-01", auto_adjust=True, progress=False)
if isinstance(dbc.columns, pd.MultiIndex):
    dbc.columns = [c[0] for c in dbc.columns]
dbc.index = pd.to_datetime(dbc.index).tz_localize(None).normalize()
c = dbc["Close"]

latest_date = c.index[-1].date()
latest_px = float(c.iloc[-1])
apr30 = c.loc[c.index <= "2026-04-30"]
apr30_date = apr30.index[-1].date()
apr30_px = float(apr30.iloc[-1])


def ret(series, end_str, lookback_months):
    end_idx = series.loc[series.index <= end_str].index[-1]
    start_target = end_idx - pd.DateOffset(months=lookback_months)
    start_idx = series.loc[series.index <= start_target].index[-1]
    return (float(series.loc[end_idx]) / float(series.loc[start_idx]) - 1) * 100


print(f"Latest close ({latest_date}): ${latest_px:.2f}")
print(f"April 30 close (signal date {apr30_date}): ${apr30_px:.2f}")
print(f"Since April 30: {(latest_px / apr30_px - 1) * 100:+.2f}%  "
      f"({(latest_date - apr30_date).days} days)")
print()
print("Trailing returns AS-OF the 4/30 signal (what the system used):")
r3 = ret(c, "2026-04-30", 3)
r6 = ret(c, "2026-04-30", 6)
r12 = ret(c, "2026-04-30", 12)
print(f"  3 month : {r3:+.2f}%")
print(f"  6 month : {r6:+.2f}%")
print(f"  12 month: {r12:+.2f}%")
print(f"  -> blended (avg of 3/6/12): {(r3 + r6 + r12) / 3:+.2f}%")
print()
print("Trailing returns AS-OF today (what the JULY rebalance will see):")
end = str(latest_date)
r3n = ret(c, end, 3)
r6n = ret(c, end, 6)
r12n = ret(c, end, 12)
print(f"  3 month : {r3n:+.2f}%")
print(f"  6 month : {r6n:+.2f}%")
print(f"  12 month: {r12n:+.2f}%")
print(f"  -> blended: {(r3n + r6n + r12n) / 3:+.2f}%")
print()
print("Last 20 daily closes:")
print(c.tail(20).to_string())
