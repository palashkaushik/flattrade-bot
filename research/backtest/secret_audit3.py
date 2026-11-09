"""Find every TRACKED file that currently contains a live secret value."""
import os
import subprocess

REPO = r"C:\Websites\FLATTRADE BOT"
vals = {}
with open(os.path.join(REPO, ".env"), encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if "=" not in line or line.strip().startswith("#"):
            continue
        k, v = line.rstrip("\n").split("=", 1)
        if k in {"FLATTRADE_API_KEY", "FLATTRADE_API_SECRET", "FLATTRADE_PASSWORD",
                 "FLATTRADE_TOKEN", "FLATTRADE_TOTP_KEY", "DISCORD_BOT_TOKEN",
                 "DISCORD_WEBHOOK_URL"} and len(v) >= 8:
            vals[k] = v

tracked = subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True, text=True).stdout.splitlines()
for k, v in vals.items():
    hits = []
    for f in tracked:
        p = os.path.join(REPO, f)
        try:
            if os.path.getsize(p) > 20_000_000:
                continue
            if v in open(p, encoding="utf-8", errors="replace").read():
                hits.append(f)
        except Exception:
            pass
    print(f"{k}: {'EXPOSED in ' + str(hits) if hits else 'not in tracked files'}")
