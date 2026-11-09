"""Check whether any .env secret value ever entered git history (local + remote)."""
import os
import subprocess

REPO = r"C:\Websites\FLATTRADE BOT"
SECRET_KEYS = ["FLATTRADE_PASSWORD", "FLATTRADE_TOKEN", "FLATTRADE_API_SECRET",
               "FLATTRADE_TOTP_KEY", "FLATTRADE_API_KEY"]

vals = {}
with open(os.path.join(REPO, ".env"), encoding="utf-8", errors="replace") as fh:
    for line in fh:
        if "=" not in line or line.strip().startswith("#"):
            continue
        k, v = line.rstrip("\n").split("=", 1)
        if k in SECRET_KEYS and len(v) >= 6:
            vals[k] = v

print("values loaded:", {k: f"{len(v)} chars" for k, v in vals.items()})
for k, v in vals.items():
    r = subprocess.run(["git", "-C", REPO, "log", "--all", "--oneline", f"-S{v}"],
                       capture_output=True, text=True)
    hits = [l for l in r.stdout.splitlines() if l.strip()]
    print(f"{k}: commits containing the value -> {len(hits)}")
    for h in hits[:3]:
        print("   ", h[:100])

# remote tree: any .env-ish file?
r = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", "origin/release"],
                   capture_output=True, text=True)
bad = [l for l in r.stdout.splitlines()
       if l.endswith(".env") or "/.env" in l or l.endswith(".pem") or l.endswith(".key")]
print("\nremote files matching .env/.pem/.key:", bad or "none")
