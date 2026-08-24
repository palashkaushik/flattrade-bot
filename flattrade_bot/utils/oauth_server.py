"""Local OAuth Redirect Server for Flattrade 1-Click Login Automation."""

import hashlib
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import httpx
from flattrade_bot.config import settings

logger = logging.getLogger("oauth_server")

LATEST_TOKEN = None


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global LATEST_TOKEN
        query = parse_qs(urlparse(self.path).query)
        code = query.get("code", [None])[0]

        if code:
            hash_input = f"{settings.FLATTRADE_API_KEY}{code}{settings.FLATTRADE_API_SECRET}".encode("utf-8")
            secret_hash = hashlib.sha256(hash_input).hexdigest()
            payload = {
                "api_key": settings.FLATTRADE_API_KEY,
                "request_code": code,
                "api_secret": secret_hash,
            }

            try:
                res = httpx.post("https://authapi.flattrade.in/trade/apitoken", json=payload, timeout=10.0)
                data = res.json()
                if data.get("stat") == "Ok" and data.get("token"):
                    LATEST_TOKEN = data["token"]
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head><title>Flattrade Bot Authenticated</title></head>
                    <body style="font-family: Arial, sans-serif; text-align: center; padding-top: 50px; background: #0f172a; color: #f8fafc;">
                        <h1 style="color: #22c55e;">✅ Flattrade Authentication Successful!</h1>
                        <p style="font-size: 1.2rem;">Live Session Token: <code>{LATEST_TOKEN[:12]}...</code></p>
                        <p style="color: #94a3b8;">Your live trading engine is now connected and active. You can close this browser tab.</p>
                    </body>
                    </html>
                    """
                else:
                    emsg = data.get("emsg") or data.get("message") or str(data)
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head><title>Flattrade Auth Error</title></head>
                    <body style="font-family: Arial, sans-serif; text-align: center; padding-top: 50px; background: #0f172a; color: #f8fafc;">
                        <h1 style="color: #ef4444;">❌ Flattrade Authentication Failed</h1>
                        <p style="font-size: 1.2rem; color: #fca5a5;">Error: {emsg}</p>
                    </body>
                    </html>
                    """
            except Exception as e:
                html = f"<h1>Error: {e}</h1>"
        else:
            html = "<h1>No authorization code parameter found in redirect URL.</h1>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, format, *args):
        pass


def run_oauth_listener(port: int = 5000) -> Optional[str]:
    """Starts single-request local OAuth server to capture code and exchange token."""
    global LATEST_TOKEN
    LATEST_TOKEN = None
    server_address = ("127.0.0.1", port)
    try:
        httpd = HTTPServer(server_address, OAuthCallbackHandler)
        logger.info(f"🌐 Local OAuth server listening on http://127.0.0.1:{port}/...")
        httpd.handle_request()  # Handles single redirect GET request then exits
        httpd.server_close()
        return LATEST_TOKEN
    except Exception as e:
        logger.error(f"Failed to start local OAuth server: {e}")
        return None
