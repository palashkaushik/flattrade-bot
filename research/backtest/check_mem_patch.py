"""Verify the categorical/self_destruct patch reproduces checkpoint results exactly."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sixyear_par as S


def main():
    done = S.load_check()
    days = S.all_days()
    idx = {d: i for i, d in enumerate(days)}
    # pick 3 computed days spread across the checkpoint
    computed = sorted(done)
    picks = [computed[0], computed[len(computed) // 2], computed[-1]]
    ok = True
    for day in picks:
        i = idx[day]
        prior = days[max(0, i - S.BUF_DAYS):i]
        t0 = time.time()
        d, res = S.worker((day, prior))
        dt = time.time() - t0
        ref = done[day]
        same = {k: [tuple(x) for x in v] == ref.get(k) for k, v in res.items()}
        good = all(same.values()) and set(res) == set(ref)
        ok = ok and good
        print(f"{day}  {dt:5.1f}s  match={good}  " + " ".join(f"{k}:{v}" for k, v in same.items()), flush=True)
    print("CONGRUENT" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
