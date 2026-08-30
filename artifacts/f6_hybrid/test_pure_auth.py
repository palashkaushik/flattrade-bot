import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import hashlib
import json
import httpx
import pyotp
import urllib.parse
from flattrade_bot.config import settings
from flattrade_bot.broker.network import _ensure_ipv4_patch

_ensure_ipv4_patch()

user_id = settings.FLATTRADE_USER_ID
pwd = settings.FLATTRADE_PASSWORD
totp_key = settings.FLATTRADE_TOTP_KEY
api_key = settings.FLATTRADE_API_KEY
api_secret = settings.FLATTRADE_API_SECRET

headers = {
    "Origin": "https://auth.flattrade.in",
    "Referer": f"https://auth.flattrade.in/?app_key={api_key}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json"
}

client = httpx.Client(base_url="https://authapi.flattrade.in", headers=headers, timeout=10.0)

# Step 1: Session
r_sess = client.post("/auth/session", json={})
sid = r_sess.text.strip().strip('"')
print(f"1. Acquired Session ID: {sid}")

# Step 2: ftauth with single SHA256
totp_code = pyotp.TOTP(totp_key).now()
pwd_hash_1 = hashlib.sha256(pwd.encode("utf-8")).hexdigest()

payload = {
    "UserName": user_id,
    "Password": pwd_hash_1,
    "PAN_DOB": totp_code,
    "App": "",
    "ClientID": "",
    "Key": "",
    "APIKey": api_key,
    "Sid": sid,
    "Rd": ""
}

r_auth = client.post("/ftauth", json=payload)
print(f"2. ftauth response (Single SHA256): {r_auth.text}")
auth_data = r_auth.json()

if auth_data.get("RedirectURL"):
    redirect_url = auth_data["RedirectURL"]
    print(f"3. Redirect URL: {redirect_url}")
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(redirect_url).query)
    code = qs.get("code", [None])[0]
    print(f"4. Request Code: {code}")
    
    if code:
        # Step 3: apitoken
        secret_hash = hashlib.sha256(f"{api_key}{code}{api_secret}".encode("utf-8")).hexdigest()
        token_payload = {
            "api_key": api_key,
            "request_code": code,
            "api_secret": secret_hash
        }
        r_token = client.post("/trade/apitoken", json=token_payload)
        print(f"5. Token response: {r_token.text}")
