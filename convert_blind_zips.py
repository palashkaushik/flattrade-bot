"""Build daily option files from vendor weekly-expiry zips.

Source: `C:/Users/user/Desktop/nifty 24 to 26/YYYYMMDD.zip` — one zip per weekly
expiry, each containing one CSV per strike (Date,Timestamp,Open,High,Low,Close,
Volume,OI,Ticker) covering the contract's full trading life.

Destination: `C:/Websites/ammu/nifty_options/YYYY/MM/nifty_options_DD_MM_YYYY.csv`
(same layout as existing ammu day files; header date,time,symbol,open,high,low,
close,oi,volume).

Day assignment rule: each trading day belongs to the NEAREST FUTURE weekly
expiry (the actively-traded contract that day) — matches how ammu's existing
2024 day files were built. Rows from later expiries for the same day are
dropped. Existing day files (2020..2024-10) are left untouched.

Symbol rewrite: zip ticker `NIFTY04JAN24CE18300` -> engine form `NIFTY04JAN2418300CE`.
Newer exports use single-letter sides, e.g. `NIFTY10MAR26P22750`; these are
normalized to `NIFTY10MAR2622750PE`.
"""

import glob
import io
import os
import re
import sys
import zipfile
from collections import defaultdict
from datetime import date, datetime, timedelta

import pandas as pd

ZIP_DIR = r"C:\Users\user\Desktop\nifty 24 to 26"
OPTS_DIR = r"C:\Websites\ammu\nifty_options"
TICKER_RE = re.compile(r"^(NIFTY\d{2}[A-Z]{3}\d{2})(CE|PE|C|P)(\d+)$")


def rewrite_symbol(ticker: str) -> str | None:
    m = TICKER_RE.match(ticker.strip())
    if not m:
        return None
    side = {"C": "CE", "P": "PE"}.get(m.group(2), m.group(2))
    return f"{m.group(1)}{m.group(3)}{side}"


def zip_expiry(path: str) -> date:
    return datetime.strptime(os.path.basename(path)[:8], "%Y%m%d").date()


def existing_days() -> set[str]:
    out = set()
    for f in glob.glob(os.path.join(OPTS_DIR, "**", "*.csv"), recursive=True):
        parts = os.path.basename(f).replace(".csv", "").split("_")
        # Existing folders use an unpadded month, while source dates are ISO.
        out.add(f"{int(parts[4]):04d}-{int(parts[3]):02d}-{int(parts[2]):02d}")
    return out


def read_zip(path: str) -> dict[str, list[tuple]]:
    """Return {day_str: [(time, symbol, o, h, l, c, oi, vol), ...]} for a zip."""
    days: dict[str, list[tuple]] = defaultdict(list)
    z = zipfile.ZipFile(path)
    for name in z.namelist():
        if not name.endswith(".csv"):
            continue
        if not name.startswith(("NIFTY", "CE", "PE")) and "_" not in name:
            continue
        fh = io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace")
        try:
            header = fh.readline()
            cols = [c.strip() for c in header.split(",")]
            if "Open" not in cols or "Ticker" not in cols:
                continue
            idx = {c: i for i, c in enumerate(cols)}
            for line in fh:
                if not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) < 9:
                    continue
                day = parts[idx["Date"]].strip()
                ts = parts[idx["Timestamp"]].strip()
                if len(ts) < 8:
                    continue
                time = ts[-8:]
                ticker = parts[idx["Ticker"]]
                sym = rewrite_symbol(ticker)
                if not sym:
                    continue
                try:
                    o = float(parts[idx["Open"]])
                    h = float(parts[idx["High"]])
                    lo = float(parts[idx["Low"]])
                    c = float(parts[idx["Close"]])
                    vol = float(parts[idx["Volume"]])
                    oi = float(parts[idx["OI"]])
                except ValueError:
                    continue
                days[day].append((time, sym, o, h, lo, c, oi, vol))
        finally:
            fh.close()
    z.close()
    return days


def main():
    zips = sorted(glob.glob(os.path.join(ZIP_DIR, "*.zip")))
    print(f"[INFO] {len(zips)} zips found", flush=True)

    existing = existing_days()
    print(f"[INFO] {len(existing)} day files already in ammu", flush=True)

    # day -> (expiry, rows) assigned to nearest future expiry
    assigned: dict[str, list[tuple]] = {}
    for zp in zips:
        expiry = zip_expiry(zp)
        days = read_zip(zp)
        print(f"[INFO] {os.path.basename(zp)[:8]} expiry {expiry} | "
              f"{len(days)} days, {sum(len(v) for v in days.values()):,} rows",
              flush=True)
        for day, rows in days.items():
            if day in existing:
                continue
            # nearest-future-expiry rule: only claim a day if no earlier
            # expiry already assigned it (zips processed in date order)
            if day not in assigned:
                if expiry >= date.fromisoformat(day):
                    assigned[day] = rows
                else:
                    continue
            else:
                continue  # later expiry must not overwrite the front contract

    print(f"\n[INFO] {len(assigned)} new day files to write", flush=True)

    written = 0
    for day, rows in sorted(assigned.items()):
        y, m, dd = day.split("-")
        out_dir = os.path.join(OPTS_DIR, y, str(int(m)))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"nifty_options_{int(dd):02d}_{int(m):02d}_{int(y):04d}.csv",
        )
        frame = pd.DataFrame(
            rows,
            columns=["time", "symbol", "open", "high", "low", "close", "oi", "volume"],
        )
        frame.insert(0, "date", day)
        frame.to_csv(out_path, index=False)
        written += 1
        if written % 20 == 0:
            print(f"  ... {written} days written", flush=True)

    print(f"\n[DONE] wrote {written} day files", flush=True)


if __name__ == "__main__":
    main()
