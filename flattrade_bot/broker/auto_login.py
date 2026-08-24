"""Flattrade Fully Automated Headless Login via Selenium + TOTP.

Drives a real Chrome session to:
1. Navigate to Flattrade OAuth login page
2. Fill in User ID, Password, and auto-generated TOTP code
3. Click "Log In"
4. Capture the redirect URL containing request_code
5. Exchange request_code for live session token
6. Return the token string

Zero-touch — no manual interaction needed.
"""

import hashlib
import logging
import os
import sys
import time
from typing import Optional
from urllib.parse import parse_qs, urlparse

import httpx
import pyotp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions

logger = logging.getLogger("auto_login")


def automated_flattrade_login(
    user_id: str,
    password: str,
    totp_key: str,
    api_key: str,
    api_secret: str,
    headless: bool = True,
    timeout: int = 30,
) -> Optional[str]:
    """Performs fully automated Flattrade login and returns session token.

    Args:
        user_id: Flattrade User ID (e.g. FZ52739)
        password: Flattrade login password (plain text)
        totp_key: Base32 TOTP secret key for 2FA
        api_key: Flattrade API Key (app_key)
        api_secret: Flattrade API Secret
        headless: Run Chrome in headless mode (no visible window)
        timeout: Max seconds to wait for each page transition

    Returns:
        Session token string, or None on failure.
    """
    # Force IPv4 socket resolution globally to match Flattrade Wall registered IPv4 address
    from flattrade_bot.broker.network import _ensure_ipv4_patch
    _ensure_ipv4_patch()

    # Generate current TOTP code
    totp = pyotp.TOTP(totp_key)
    totp_code = totp.now()
    logger.info(f"Generated TOTP code: {totp_code}")

    # Setup Chrome with forced IPv4 networking
    chrome_options = ChromeOptions()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,900")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        wait = WebDriverWait(driver, timeout)

        auth_url = f"https://auth.flattrade.in/?app_key={api_key}"
        logger.info(f"Navigating to: {auth_url}")
        driver.get(auth_url)

        # Wait for Vue.js SPA to render
        time.sleep(3)

        # ─── Step 1: Enter User ID ───
        logger.info("Filling User ID...")
        uid_field = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="User ID"]'))
        )
        uid_field.clear()
        uid_field.send_keys(user_id)
        logger.info(f"Entered User ID: {user_id}")

        # ─── Step 2: Enter Password ───
        logger.info("Filling Password...")
        pwd_field = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="Password"]')
        pwd_field.clear()
        pwd_field.send_keys(password)
        logger.info("Entered Password")

        # ─── Step 3: Enter TOTP ───
        logger.info("Filling TOTP...")
        totp_field = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="OTP / TOTP"]')
        totp_field.clear()
        totp_field.send_keys(totp_code)
        logger.info(f"Entered TOTP code: {totp_code}")

        # ─── Step 4: Click "Log In" button ───
        logger.info("Clicking Log In button...")
        login_btn = driver.find_element(By.CSS_SELECTOR, "button.shine-button")
        login_btn.click()
        logger.info("Clicked Log In")

        # ─── Step 5: Wait for redirect to capture request_code ───
        logger.info("Waiting for redirect with request_code...")
        request_code = None

        for attempt in range(timeout * 2):  # Poll every 0.5s
            current_url = driver.current_url
            if "code=" in current_url:
                parsed = urlparse(current_url)
                code_vals = parse_qs(parsed.query).get("code", [])
                if code_vals:
                    request_code = code_vals[0]
                    logger.info(f"Captured request_code: {request_code}")
                    break

            if attempt > 0 and attempt % 10 == 0:
                try:
                    snack = driver.find_element(By.CSS_SELECTOR, ".v-snack__content p")
                    if snack.text.strip():
                        logger.warning(f"Page message: {snack.text.strip()}")
                except Exception:
                    pass

            time.sleep(0.5)

        if not request_code:
            try:
                body_text = driver.find_element(By.TAG_NAME, "body").text
                logger.error(f"No request_code captured. Page text: {body_text[:300]}")
            except Exception:
                pass
            driver.save_screenshot("auto_login_error.png")
            logger.info("Saved error screenshot to auto_login_error.png")
            return None

        # ─── Step 6: Exchange request_code for session token over IPv4 ───
        logger.info("Exchanging request_code for session token over forced IPv4...")
        hash_input = f"{api_key}{request_code}{api_secret}"
        secret_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        token_payload = {
            "api_key": api_key,
            "request_code": request_code,
            "api_secret": secret_hash,
        }

        # Send via httpx with forced IPv4 socket
        res = httpx.post(
            "https://authapi.flattrade.in/trade/apitoken",
            json=token_payload,
            timeout=15.0,
        )
        data = res.json()
        logger.info(f"Flattrade token response: {res.text[:200]}")

        if (data.get("stat") == "Ok" or data.get("status") == "Ok") and data.get("token"):
            token = data["token"]
            logger.info(f"🎉 Session token acquired via IPv4: {token[:12]}...")
            return token
        else:
            emsg = data.get("emsg", str(data))
            logger.error(f"Token exchange failed: {emsg}")
            return None

    except Exception as e:
        logger.error(f"Auto-login error: {e}")
        if driver:
            try:
                driver.save_screenshot("auto_login_error.png")
                logger.info("Saved error screenshot to auto_login_error.png")
            except Exception:
                pass
        return None
    finally:
        if driver:
            driver.quit()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys.stdout.reconfigure(encoding="utf-8")

    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    password = os.getenv("FLATTRADE_PASSWORD", "")
    if not password:
        print("ERROR: Set FLATTRADE_PASSWORD in .env for fully automated login.")
        sys.exit(1)

    token = automated_flattrade_login(
        user_id=os.getenv("FLATTRADE_USER_ID", ""),
        password=password,
        totp_key=os.getenv("FLATTRADE_TOTP_KEY", ""),
        api_key=os.getenv("FLATTRADE_API_KEY", ""),
        api_secret=os.getenv("FLATTRADE_API_SECRET", ""),
        headless=False,
    )

    if token:
        print(f"\nSUCCESS: Token = {token[:16]}...")
    else:
        print("\nFAILED: Could not obtain token.")
