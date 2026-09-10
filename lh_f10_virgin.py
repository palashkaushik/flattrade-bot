"""F10 virgin-CPR pre-pass — build the 7-year virgin map for the parquet run.

Virgin CPR = the 3-line CPR (CPR/BC/TC, half-up) of the FIRST day each weekly
contract family appears in the parquet (contract's own listed history — matches
Upstox charts). A line is ALIVE for a given later day if no bar of that
contract ever traded through it (low <= line <= high) between its birth day
(exclusive) and that day (exclusive). Today's walk happens at replay time.

Output: lh_f10_virgin_map.json  {sym: {day: {vals: [cpr,bc,tc], alive: [0/1 x3]}}}
Only days where the contract is day-eligible AND >=2nd day of its life.
"""
import sys, json, re
sys.path.insert(0, r"C:\Websites\FLATTRADE BOT")
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

PARQ = r"C:\Users\user\AppData\Local\Temp\opencode\data\nifty50_options_master_v2.parquet"
OUT = r"C:\Websites\FLATTRADE BOT\lh_f10_virgin_map.json"
T0 = 555

def r2(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

print("loading parquet (day, symbol, minute, OHLC)...")
pq = pd.read_parquet(PARQ, columns=["day", "minute", "symbol", "open", "high", "low", "close"])
pq["day"] = pd.to_datetime(pq["day"])
print(f"{len(pq):,} rows; grouping...")

out = {}
sym_first = {}      # sym -> first day ts
sym_days = pq.groupby("symbol").day.min()
for sym, d0 in sym_days.items():
    sym_first[sym] = d0

# process each symbol: birth-day CPR + touch walk
for sym, g in pq.groupby("symbol", sort=False):
    g = g.sort_values(["day", "minute"])
    d0 = g.day.iloc[0]
    birth = g[g.day == d0]
    if not len(birth):
        continue
    h = float(birth.high.max()); l = float(birth.low.min()); c = float(birth.close.iloc[-1])
    cpr = r2((h + l + c) / 3.0); bc = r2((h + l) / 2.0); tc = r2(2.0 * cpr - bc)
    vals = [cpr, bc, tc]
    alive = [True, True, True]
    per_day = {}
    # walk days after birth
    gd = g[g.day > d0]
    if len(gd):
        for day, gg in gd.groupby("day"):
            # record state at day open (alive BEFORE today's bars)
            per_day[str(day.date())] = dict(vals=vals, alive=[int(a) for a in alive])
            # walk today's bars
            lows = gg.low.values; highs = gg.high.values
            for li in range(3):
                if not alive[li]:
                    continue
                v = vals[li]
                if ((lows <= v) & (highs >= v)).any():
                    alive[li] = False
    if per_day:
        out[sym] = per_day

with open(OUT, "w") as f:
    json.dump(out, f)
print(f"virgin map: {len(out)} syms -> {OUT}")
