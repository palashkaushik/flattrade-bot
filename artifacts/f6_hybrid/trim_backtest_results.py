"""trim_backtest_results.py — keep only the leaderboard, delete heavy data.

For marni-style files (dict keyed by param-combo, each = {stats, trades}):
  - group combos by category (first '|' token of the key, e.g. 1m/2m/3m/5m/combined)
  - rank combos within each category by net_rs (best first)
  - keep top-N per category (default 5)
  - drop the heavy 'trades' list (keep only the lean 'stats' summary)

For engine-style files (master_phase5d_*, *_comparison.json) the payload is
already leaderboard-only; we just slice the top-N and report.

Writes <name>_trimmed.json, then (with --delete) removes the original.
"""
import json
import os
import sys
import glob

TOP_N = 5


def trim_marni_style(d, top_n):
    """d: dict combo->{'stats':..., 'trades':[...]}. Returns trimmed dict."""
    cats = {}
    for key, val in d.items():
        cat = key.split("|")[0]
        cats.setdefault(cat, []).append((key, val))
    out = {}
    for cat, items in cats.items():
        items.sort(key=lambda kv: kv[1].get("stats", {}).get("net_rs", float("-inf")),
                   reverse=True)
        for key, val in items[:top_n]:
            stats = val.get("stats", {})
            out[key] = {"stats": stats}  # drop 'trades' (the heavy part)
    return out


def trim_engine_style(d, top_n):
    out = dict(d)
    for k in ("results", "nw_top10", "wf_oos_top10", "nw_top", "wf_oos_top"):
        if isinstance(d.get(k), list):
            out[k] = d[k][:top_n]
    return out


def is_marni_style(d):
    if not isinstance(d, dict) or not d:
        return False
    v = next(iter(d.values()))
    return isinstance(v, dict) and "stats" in v and "trades" in v


def process(path, top_n, delete):
    size_before = os.path.getsize(path)
    d = json.load(open(path, "r", encoding="utf-8"))
    if is_marni_style(d):
        trimmed = trim_marni_style(d, top_n)
        kind = "marni"
    elif isinstance(d, dict):
        trimmed = trim_engine_style(d, top_n)
        kind = "engine"
    else:
        print(f"  SKIP (unknown structure): {path}")
        return
    base, ext = os.path.splitext(path)
    out_path = base + "_trimmed" + ext
    json.dump(trimmed, open(out_path, "w", encoding="utf-8"), indent=2, default=float)
    size_after = os.path.getsize(out_path)
    freed = size_before - size_after
    print(f"  [{kind}] {os.path.basename(path)}: {size_before/1e6:.1f}MB -> "
          f"{size_after/1e6:.2f}MB (freed {freed/1e6:.1f}MB)")
    if delete:
        os.remove(path)
        print(f"        deleted original")


def main():
    paths = []
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        paths += glob.glob(arg)
    paths = sorted(set(paths))
    top_n = TOP_N
    delete = "--keep" not in sys.argv
    total_freed = 0
    for p in paths:
        if p.endswith("_trimmed.json"):
            continue
        print(f"Processing {p}")
        process(p, top_n, delete)
    print("Done.")


if __name__ == "__main__":
    main()
