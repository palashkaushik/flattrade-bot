"""Audit every .env value against git history; locate the files holding them."""
import os
import subprocess

REPO = r"C:\Websites\FLATTRADE BOT"
SKIP = {"STRATEGY_MODE"}
vals = {}
with open(os.path.join(REPO, ".env"), encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if "=" not in line or line.strip().startswith("#"):
            continue
        k, v = line.rstrip("\n").split("=", 1)
        if k not in SKIP and len(v) >= 8:
            vals[k] = v

for k, v in vals.items():
    r = subprocess.run(["git", "-C", REPO, "log", "--all", "--oneline", f"-S{v}"],
                       capture_output=True, text=True)
    hits = [l for l in r.stdout.splitlines() if l.strip()]
    if not hits:
        print(f"{k}: clean")
        continue
    print(f"{k}: {len(hits)} commits")
    # locate the files in the newest hit
    commit = hits[0].split()[0]
    g = subprocess.run(["git", "-C", REPO, "grep", "-l", v, commit],
                       capture_output=True, text=True)
    files = [l.split(":", 1)[1] for l in g.stdout.splitlines() if ":" in l]
    print("   newest commit:", hits[0][:80])
    print("   files:", sorted(set(files))[:5])
