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

logger = logging.getLogger("auto_login")


def automated_flattrade_login_playwright(
    user_id: str,
    password: str,
    totp_key: str,
    api_key: str,
    api_secret: str,
    timeout: int = 30,
) -> Optional[str]:
    """100% Autonomous zero-touch headless login via Playwright."""
    from flattrade_bot.broker.network import _ensure_ipv4_patch
    _ensure_ipv4_patch()

    totp = pyotp.TOTP(totp_key)
    totp_code = totp.now()
    logger.info(f"Generated TOTP for Playwright auto-login: {totp_code}")

    auth_url = f"https://auth.flattrade.in/?app_key={api_key}"
    request_code = None

    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            page.goto(auth_url, timeout=timeout * 1000)
            page.wait_for_timeout(2000)

            # Step 1: User ID
            page.fill('input[placeholder="User ID"]', user_id)
            # Step 2: Password
            page.fill('input[placeholder="Password"]', password)
            # Step 3: TOTP
            page.fill('input[placeholder="OTP / TOTP"]', totp_code)
            # Step 4: Submit
            page.click("button.shine-button")

            # Step 5: Capture redirect code
            for _ in range(timeout * 2):
                curr = page.url
                if "code=" in curr:
                    parsed = urlparse(curr)
                    c_vals = parse_qs(parsed.query).get("code", [])
                    if c_vals:
                        request_code = c_vals[0]
                        logger.info(f"Playwright captured request_code: {request_code}")
                        break
                page.wait_for_timeout(500)

            browser.close()

        if request_code:
            # Exchange for live token
            url = "https://authapi.flattrade.in/trade/apitoken"
            hash_input = f"{api_key}{request_code}{api_secret}"
            secret_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
            payload = {
                "api_key": api_key,
                "request_code": request_code,
                "api_secret": secret_hash,
            }
            res = httpx.post(url, json=payload, timeout=10.0)
            data = res.json()
            if data.get("stat") == "Ok" or data.get("status") == "Ok":
                token = data.get("token")
                logger.info(f"✅ Playwright Live Session Token Acquired: {token[:8]}...")
                return token
            else:
                logger.error(f"❌ Token exchange failed: {data}")
    except Exception as e:
        logger.warning(f"Playwright auto-login attempt failed: {e}")

def automated_flattrade_login_selenium(
    user_id: str,
    password: str,
    totp_key: str,
    api_key: str,
    api_secret: str,
    headless: bool = True,
    timeout: int = 30,
) -> Optional[str]:
    from flattrade_bot.broker.network import _ensure_ipv4_patch
    _ensure_ipv4_patch()

    totp = pyotp.TOTP(totp_key)
    totp_code = totp.now()
    logger.info(f"Generated TOTP code for Selenium: {totp_code}")

    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions

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
        time.sleep(3)

        uid_field = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'input[placeholder="User ID"]'))
        )
        uid_field.clear()
        uid_field.send_keys(user_id)

        pwd_field = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="Password"]')
        pwd_field.clear()
        pwd_field.send_keys(password)

        totp_field = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="OTP / TOTP"]')
        totp_field.clear()
        totp_field.send_keys(totp_code)

        login_btn = driver.find_element(By.CSS_SELECTOR, "button.shine-button")
        login_btn.click()

        request_code = None
        for _ in range(timeout * 2):
            curr_url = driver.current_url
            if "code=" in curr_url:
                parsed = urlparse(curr_url)
                code_vals = parse_qs(parsed.query).get("code", [])
                if code_vals:
                    request_code = code_vals[0]
                    logger.info(f"Selenium captured request_code: {request_code}")
                    break
            time.sleep(0.5)

        if not request_code:
            return None

        url = "https://authapi.flattrade.in/trade/apitoken"
        hash_input = f"{api_key}{request_code}{api_secret}"
        secret_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        token_payload = {
            "api_key": api_key,
            "request_code": request_code,
            "api_secret": secret_hash,
        }
        res = httpx.post(url, json=token_payload, timeout=15.0)
        data = res.json()
        if (data.get("stat") == "Ok" or data.get("status") == "Ok") and data.get("token"):
            return data["token"]
        return None

    except Exception as e:
        logger.warning(f"Selenium auto-login attempt: {e}")
        return None
    finally:
        if driver:
            driver.quit()


def save_token_to_env(token: str):
    from flattrade_bot.config import settings
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    lines = []
    found = False
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.startswith("FLATTRADE_TOKEN="):
                lines[i] = f"FLATTRADE_TOKEN={token}"
                found = True
                break
    if not found:
        lines.append(f"FLATTRADE_TOKEN={token}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"✅ Saved live session token to {env_file}")


def automated_flattrade_login(
    user_id: str,
    password: str,
    totp_key: str,
    api_key: str,
    api_secret: str,
    headless: bool = True,
    timeout: int = 30,
) -> Optional[str]:
    """100% Zero-Touch Automated Login: Tries Playwright first, then Selenium."""
    # 1. Try Playwright
    token = automated_flattrade_login_playwright(
        user_id=user_id,
        password=password,
        totp_key=totp_key,
        api_key=api_key,
        api_secret=api_secret,
        timeout=timeout,
    )
    if token:
        save_token_to_env(token)
        return token

    # 2. Try Selenium
    token = automated_flattrade_login_selenium(
        user_id=user_id,
        password=password,
        totp_key=totp_key,
        api_key=api_key,
        api_secret=api_secret,
        headless=headless,
        timeout=timeout,
    )
    if token:
        save_token_to_env(token)
        return token

    logger.error("❌ Both Playwright and Selenium headless login attempts failed.")
    return None
