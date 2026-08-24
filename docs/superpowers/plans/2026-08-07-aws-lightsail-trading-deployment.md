# AWS Lightsail Trading Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the Flattrade trading bot on an AWS Lightsail Mumbai Ubuntu server that authenticates after 05:00 IST, starts unattended before the 09:15 IST trading window, squares off positions at 15:00 IST, and fails closed under process, network, credential, or broker-state failures.

**Architecture:** Use an AWS Lightsail Linux/Unix 2 GB, 2 vCPU instance in Mumbai with one persistent public IPv4. Run the bot as a least-privilege `flattrade` systemd service, use separate systemd timers for startup, independent broker-side square-off, health checking, and final stop, and keep secrets in a root-owned runtime environment file plus a mode-0600 session-token file. The bot will reconcile remote Flattrade state before enabling entries and will never infer that an accepted order was filled without OrderBook/TradeBook confirmation.

**Tech Stack:** Python 3.11+, Ubuntu 24.04 LTS x86_64, AWS Lightsail, systemd, OpenSSH, UFW, Selenium headless Chrome, Flattrade Pi API, `httpx`, `pyotp`, `python-dotenv`, Rich, Discord webhook notifications, direct-function regression tests, `pip-audit`, and the existing security scanner.

## Global Constraints

- AWS Lightsail Mumbai is the deployment target; do not switch to EC2, ARM, Windows, RDP, or an IPv6-only instance for this rollout.
- Use the Linux/Unix 2 GB, 2 vCPU, 60 GB SSD Lightsail bundle with a persistent static public IPv4; expected base price is approximately `$12/month` before taxes and exchange fees.
- Flattrade API traffic must leave through the whitelisted static IPv4; verify with `curl -4 ifconfig.me` from the server before requesting the API whitelist update.
- The server must run Ubuntu 24.04 LTS x86_64 and the bot must run as the unprivileged `flattrade` user, never as root.
- The server must expose SSH only to the administrator's current `/32` address or a private VPN; no public HTTP, HTTPS, RDP, Jupyter, or debug port is allowed.
- Never commit `.env`, API keys, API secrets, passwords, TOTP seeds, session tokens, SSH private keys, AWS credentials, or raw broker response bodies containing sensitive account data.
- Never log TOTP values, passwords, request codes, session tokens, API keys, full authorization URLs, or raw request bodies.
- The live service must use `--auto-login --live-orders` only after all safety gates pass; local debugging may use `--no-headless`, but the VPS must remain headless.
- The process should start at 09:05 IST to allow login and historical warmup; the entry gate should be configuration-driven and default to 09:15 IST for this deployment.
- The EOD exit is due at 15:00 IST; the process stop at 15:05 IST is only a lifecycle failsafe and is never the primary square-off mechanism.
- Systemd calendar timers must use `Persistent=false`; a server that was offline at the start time must not start trading late after it comes back.
- Automatic restart is permitted only after remote-position reconciliation and the independent square-off path are implemented and tested.
- A dedicated Flattrade account or account scope with no unrelated open NFO positions is required for unattended live operation; otherwise the bot must enter safe halt when it sees unknown remote exposure.
- No implementation can guarantee that a broker, network, cloud provider, or market will never fail. The security target is defense in depth, fail-closed behavior, rapid detection, and a documented manual emergency procedure.
- Every new direct-function test module must expose `run_all()`; synchronous tests are called directly and asynchronous tests are executed with `asyncio.run()` so the repository remains runnable without pytest.

---

## Current Code Boundaries

The existing graph and source structure already provide the following seams:

- `flattrade_bot/main.py:78-607` owns `TradingEngine`, authentication, warmup, polling, dashboard rendering, signal routing, and lifecycle shutdown.
- `flattrade_bot/execution.py:13-221` owns risk-gated entry, fill confirmation, local position state, exits, P&L, and notifications.
- `flattrade_bot/broker/client.py:12-159` owns Flattrade PlaceOrder, PositionBook, OrderBook, CancelOrder, and TradeBook calls.
- `flattrade_bot/broker/history.py` owns historical candles, live quotes, and contract resolution.
- `flattrade_bot/broker/auto_login.py:33-196` performs Selenium/TOTP login and token exchange.
- `flattrade_bot/risk/manager.py:21-84` currently hardcodes a 09:20-15:00 session window and enforces daily/consecutive-loss limits.
- `flattrade_bot/utils/lifecycle.py:1-143` already handles local terminal shutdown and parent-process monitoring; the VPS deployment will use systemd as the process owner and retain this code as a clean signal handler.
- `test_execution.py`, `test_dashboard.py`, and `test_lifecycle.py` use direct test functions rather than requiring pytest; new tests should follow this convention.

## Threat Model

| Threat | Required mitigation | Verification |
|---|---|---|
| SSH brute force or credential theft | Ed25519 keys, no password/root login, provider firewall plus UFW, admin `/32` or VPN only | `sshd -T`, `ufw status verbose`, Lightsail firewall review |
| Flattrade credentials or TOTP seed leak | Root-owned 0640 env file readable only by service group; no secrets in Git, logs, shell history, or process arguments beyond required flags | secret scan, file-mode test, log redaction test |
| Whitelisted IP changes | Persistent Lightsail static IPv4, `curl -4 ifconfig.me` verification, no instance replacement without rechecking | deployment verification script |
| Bot restart with an existing position | Startup PositionBook reconciliation; block new entries if remote exposure is nonzero or ambiguous | reconciliation tests and staged restart drill |
| Local state lost after crash | Independent square-off reads broker positions, not local `TradeExecutor.position` | square-off integration test |
| Order accepted but rejected/unfilled | OrderBook and TradeBook fill confirmation; no local position until confirmed | rejected-order regression test and live read-only check |
| Service hang while process remains alive | Heartbeat file plus systemd health timer; stale heartbeat triggers safe restart and alert | healthcheck tests and stale-heartbeat drill |
| Late start after server outage | `Persistent=false`, holiday guard, no catch-up timer behavior | timer configuration review and outage simulation |
| Clock drift or incorrect timezone | `Asia/Kolkata`, chrony/systemd time sync, timer `After=time-sync.target` | `timedatectl`, `chronyc tracking`, `systemd-analyze calendar` |
| Unknown NFO positions in the account | Dedicated account scope; safe halt on non-bot or non-NIFTY exposure | remote-state tests |
| Malicious or vulnerable dependency | Pinned runtime dependency lock, `pip-audit`, update review before deployment | dependency scan in release checklist |
| Provider compromise or account takeover | AWS account MFA, no access keys on VPS, least-privilege Lightsail/IAM operator, billing alerts | AWS account checklist |

---

### Task 1: Make Session Timing Configuration-Driven

**Files:**
- Modify: `flattrade_bot/config.py`
- Modify: `flattrade_bot/risk/manager.py`
- Create: `test_config.py`
- Test: `test_session_opt.py`
- Documentation: `README.md`

**Interfaces:**
- Produces `parse_hhmm(value: str) -> int` in `flattrade_bot.config`, returning minutes since midnight and raising `ValueError` for malformed or out-of-range values.
- `RiskManager.__init__` consumes optional `session_start_min` and `session_end_min`; when omitted, it parses `settings.SESSION_START` and `settings.SESSION_END` instead of using hardcoded `09:20` and `15:00` values.

- [ ] **Step 1: Write failing timing tests.**

```python
from flattrade_bot.config import parse_hhmm
from flattrade_bot.risk.manager import RiskManager


def test_parse_hhmm_returns_minutes_since_midnight():
    assert parse_hhmm("09:15") == 555
    assert parse_hhmm("15:00") == 900


def test_parse_hhmm_rejects_invalid_clock_values():
    for value in ("9:15", "24:00", "12:60", "", "abc"):
        try:
            parse_hhmm(value)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {value!r}")


def test_default_risk_window_uses_settings():
    risk = RiskManager()
    assert risk.is_session_active(555) is True
    assert risk.is_session_active(900) is False
```

- [ ] **Step 2: Run the focused tests and confirm the current hardcoded defaults fail.**

Run: `python -c "import test_config; test_config.test_parse_hhmm_returns_minutes_since_midnight(); test_config.test_parse_hhmm_rejects_invalid_clock_values(); test_config.test_default_risk_window_uses_settings()"`

Expected: FAIL because `parse_hhmm` is missing and `RiskManager` defaults to 09:20.

- [ ] **Step 3: Add the configuration parser and environment-backed defaults.**

Use these defaults in `Settings`:

```python
SESSION_START: str = os.getenv("SESSION_START", "09:15")
SESSION_END: str = os.getenv("SESSION_END", "15:00")
TRADING_TIMEZONE: str = os.getenv("TRADING_TIMEZONE", "Asia/Kolkata")
SESSION_TOKEN_PATH: Path = Path(
    os.getenv("SESSION_TOKEN_PATH", "/run/flattrade-bot/session.token")
)
```

`parse_hhmm` must validate exactly two decimal hour digits, a colon, two decimal minute digits, `0 <= hour <= 23`, and `0 <= minute <= 59`.

- [ ] **Step 4: Update `RiskManager` to use parsed settings while preserving explicit test overrides.**

The constructor contract must become:

```python
def __init__(
    self,
    max_daily_loss_rs: float = settings.MAX_DAILY_LOSS_RS,
    consecutive_loss_limit: int = settings.CONSECUTIVE_LOSS_LIMIT,
    session_start_min: int | None = None,
    session_end_min: int | None = None,
):
    self.session_start_min = (
        parse_hhmm(settings.SESSION_START)
        if session_start_min is None
        else session_start_min
    )
    self.session_end_min = (
        parse_hhmm(settings.SESSION_END)
        if session_end_min is None
        else session_end_min
    )
```

- [ ] **Step 5: Run the focused tests and the existing session tests.**

Run: `python -c "import test_config; test_config.test_parse_hhmm_returns_minutes_since_midnight(); test_config.test_parse_hhmm_rejects_invalid_clock_values(); test_config.test_default_risk_window_uses_settings(); import test_session_opt; test_session_opt.test_cutoff(); print('TIMING_TESTS_PASS')"`

Expected: `TIMING_TESTS_PASS`.

- [ ] **Step 6: Document that the service starts at 09:05 for warmup while entries are gated from 09:15 to 15:00.**

- [ ] **Step 7: Commit.**

```bash
git add flattrade_bot/config.py flattrade_bot/risk/manager.py test_config.py test_session_opt.py README.md
git commit -m "Make trading session timing configurable"
```

### Task 2: Secure Session Token Storage and Login Logging

**Files:**
- Create: `flattrade_bot/broker/session_store.py`
- Modify: `flattrade_bot/broker/auto_login.py`
- Modify: `flattrade_bot/main.py`
- Modify: `flattrade_bot/config.py`
- Test: `test_session_store.py`
- Test: `test_auto_login_security.py`

**Interfaces:**
- `store_token(token: str, path: Path) -> None` writes a regular file with mode `0600`, creating parent directories with mode `0700`.
- `load_token(path: Path) -> str` returns a non-empty token only when the file is regular, not a symlink, and has no group/other permission bits; otherwise it raises `RuntimeError`.
- `clear_token(path: Path) -> None` removes the token file if it exists.

- [ ] **Step 1: Write token-store tests.**

```python
from pathlib import Path

from flattrade_bot.broker.session_store import clear_token, load_token, store_token


def test_store_and_load_token_with_private_permissions(tmp_path: Path):
    token_path = tmp_path / "runtime" / "session.token"
    store_token("daily-token", token_path)
    assert load_token(token_path) == "daily-token"
    assert token_path.stat().st_mode & 0o077 == 0


def test_load_token_rejects_symlink(tmp_path: Path):
    real_path = tmp_path / "real.token"
    link_path = tmp_path / "session.token"
    store_token("daily-token", real_path)
    link_path.symlink_to(real_path)
    try:
        load_token(link_path)
    except RuntimeError:
        pass
    else:
        raise AssertionError("symlink token path must be rejected")


def test_clear_token_removes_token(tmp_path: Path):
    token_path = tmp_path / "session.token"
    store_token("daily-token", token_path)
    clear_token(token_path)
    assert not token_path.exists()
```

- [ ] **Step 2: Run the tests and confirm the module is missing.**

Run: `python -c "import test_session_store; test_session_store.test_store_and_load_token_with_private_permissions(__import__('pathlib').Path(__import__('tempfile').mkdtemp()))"`

Expected: FAIL because the session-store module is not implemented.

- [ ] **Step 3: Implement atomic token storage without logging token contents.**

Use `os.open` with `O_CREAT | O_TRUNC | O_WRONLY`, mode `0o600`, write the token with UTF-8, flush and `os.fsync`, then `os.replace` a temporary file into place. Reject symlinks with `path.is_symlink()` before opening. Do not use a shell command or environment interpolation to write the token.

- [ ] **Step 4: Remove sensitive logging from `auto_login.py`.**

Replace logs that print the TOTP code, request code, or token prefix with fixed messages such as `Generated TOTP code`, `Captured authorization redirect`, and `Session token acquired`. Never log the redirect URL or response body from the token exchange.

- [ ] **Step 5: Store the daily token after successful authentication.**

After `TradingEngine.initialize` obtains a token, call `store_token(token, settings.SESSION_TOKEN_PATH)`. The token path is `/run/flattrade-bot/session.token` in production and a temporary path in tests. Clear the file in `finally` when the main service exits; the square-off command first loads the file and falls back to a fresh automated login if the file is absent. The systemd runtime directory is tmpfs and is deleted on reboot.

- [ ] **Step 6: Add security tests for login logging.**

Use a fake logger/caplog-style handler or a `StringIO` handler around a mocked login flow. Assert that a known TOTP, request code, and token do not appear in emitted log text.

- [ ] **Step 7: Run tests and a secret scan.**

Run: `python -c "import test_session_store; test_session_store.test_store_and_load_token_with_private_permissions(__import__('pathlib').Path(__import__('tempfile').mkdtemp())); import test_auto_login_security; test_auto_login_security.test_login_logs_do_not_contain_secrets(); print('SESSION_SECURITY_TESTS_PASS')"`

Run: `npx @claude-flow/cli security validate --check secrets`

Expected: `SESSION_SECURITY_TESTS_PASS` and no hardcoded secrets.

- [ ] **Step 8: Commit.**

```bash
git add flattrade_bot/broker/session_store.py flattrade_bot/broker/auto_login.py flattrade_bot/main.py flattrade_bot/config.py test_session_store.py test_auto_login_security.py
git commit -m "Protect unattended session credentials"
```

### Task 3: Reconcile Broker State Before Any New Entry

**Files:**
- Modify: `flattrade_bot/broker/client.py`
- Modify: `flattrade_bot/execution.py`
- Modify: `flattrade_bot/main.py`
- Modify: `flattrade_bot/utils/discord.py`
- Modify: `test_execution.py`
- Create: `test_reconciliation.py`

**Interfaces:**
- Change `FlattradeClient.place_market_order` to accept `remarks: str = "QuadRotation_Bot"` and pass it to the Flattrade API payload.
- Add `TradeExecutor.reconcile_remote_state(*, symbol_prefix: str = "NIFTY") -> dict[str, Any]`.
- Add `DiscordNotifier.notify_system_event(title: str, details: dict[str, Any], severity: str = "warning") -> None` with redaction of tokens, credentials, and raw broker bodies.
- `test_reconciliation.py` defines `make_executor(positions, orders, positions_response=None, orders_response=None) -> tuple[TradeExecutor, FakeClient]` so each async test can construct isolated state without a live client.

- [ ] **Step 1: Write failing reconciliation tests.**

Cover these exact cases:

```python
async def test_reconciliation_allows_entry_when_position_and_orders_are_empty():
    executor, _client = make_executor(positions=[], orders=[])
    result = await executor.reconcile_remote_state()
    assert result["safe_to_trade"] is True


async def test_reconciliation_blocks_when_nifty_position_exists():
    executor, _client = make_executor(
        positions=[{"exch": "NFO", "tsym": "NIFTY11AUG26C24450", "netqty": "65"}],
        orders=[],
    )
    result = await executor.reconcile_remote_state()
    assert result["safe_to_trade"] is False
    assert result["reason"] == "REMOTE_POSITION_EXISTS"


async def test_reconciliation_blocks_when_broker_api_returns_error():
    executor, _client = make_executor(
        positions=[],
        orders=[],
        positions_response={"stat": "Not_Ok", "emsg": "Session Expired"},
    )
    result = await executor.reconcile_remote_state()
    assert result["safe_to_trade"] is False
    assert result["reason"] == "BROKER_STATE_UNAVAILABLE"


async def test_reconciliation_blocks_unknown_non_nifty_nfo_exposure():
    executor, _client = make_executor(
        positions=[{"exch": "NFO", "tsym": "BANKNIFTY11AUG26C50000", "netqty": "15"}],
        orders=[],
    )
    result = await executor.reconcile_remote_state()
    assert result["safe_to_trade"] is False
    assert result["reason"] == "UNKNOWN_REMOTE_EXPOSURE"
```

The fake client must return the same list/dict response shapes used by Flattrade: PositionBook and OrderBook can be arrays on success and dictionaries on failure.

The module must provide `run_all()` that invokes each async test with `asyncio.run()` and constructs an isolated fake client, risk manager, notifier, and executor for each scenario.

- [ ] **Step 2: Run the tests and confirm the reconciliation interface is missing.**

Run: `python -c "import test_reconciliation; test_reconciliation.test_reconciliation_allows_entry_when_position_and_orders_are_empty()"`

Expected: FAIL because `reconcile_remote_state` is missing.

- [ ] **Step 3: Add a single normalization path for broker list responses.**

Normalize a successful list directly, normalize a dictionary only from its known `orders` or `positions` key, and treat every other shape or `stat=Not_Ok` as unavailable. Never treat an API error as an empty account.

- [ ] **Step 4: Implement fail-closed reconciliation.**

`reconcile_remote_state` must:

1. Call `get_positions` and `get_order_book`.
2. Reject any nonzero NFO position that is not a NIFTY contract.
3. Reject any nonzero NIFTY position because local SL/target state cannot be safely reconstructed after a restart.
4. Reject any open or pending NIFTY order without a `QuadRotation_Bot` remarks prefix.
5. Cancel only bot-tagged pending orders using `cancel_order`.
6. Return a redacted summary containing symbols, net quantities, statuses, and a machine-readable reason.
7. Return `safe_to_trade=True` only when both broker calls succeed and no remote exposure remains.

- [ ] **Step 5: Gate the live engine before warmup and signal processing.**

After authentication and before live warmup, call reconciliation. When unsafe, set both `TradingEngine.live_orders` and `TradeExecutor.live_orders` to `False`, display `SAFE HALT - REMOTE STATE`, emit a Discord alert, and continue only in read-only monitoring mode. The engine must not call `place_market_order` after an unsafe result.

- [ ] **Step 6: Add unique redacted order tags.**

Use a tag such as `QuadRotation_Bot:20260807:CE:1m:7f3a2c1d` in the Flattrade `remarks` field, replacing the final eight-character example with the first eight characters of a new `uuid4()` for every order. Logs may include the tag and order ID, but never the session token or full request body.

- [ ] **Step 7: Run the reconciliation and existing execution tests.**

Run: `python -c "import test_reconciliation; test_reconciliation.run_all(); import test_execution; test_execution.test_trade_executor_does_not_create_position_after_orderbook_rejection(); print('RECONCILIATION_TESTS_PASS')"`

Expected: `RECONCILIATION_TESTS_PASS` and no fake client order submission in unsafe cases.

- [ ] **Step 8: Commit.**

```bash
git add flattrade_bot/broker/client.py flattrade_bot/execution.py flattrade_bot/main.py flattrade_bot/utils/discord.py test_execution.py test_reconciliation.py
git commit -m "Fail closed on remote broker exposure"
```

### Task 4: Build an Independent Broker-Position Square-Off Command

**Files:**
- Create: `flattrade_bot/squareoff.py`
- Modify: `flattrade_bot/execution.py`
- Modify: `flattrade_bot/broker/client.py`
- Test: `test_squareoff.py`
- Documentation: `README.md`

**Interfaces:**
- Add `async def square_off_nifty_positions(client: FlattradeClient, now: datetime) -> dict[str, Any]`.
- Add a CLI entry point: `python -m flattrade_bot.squareoff`.
- The CLI returns exit code `0` only when every NIFTY NFO position is confirmed flat or no positions exist; it returns nonzero for unknown exposure, broker failure, rejected exits, or incomplete verification.
- `test_squareoff.py` defines `make_client(positions, quotes, remaining_positions=None, order_status="COMPLETE") -> FakeClient` so each test controls both the initial and post-exit PositionBook responses.

- [ ] **Step 1: Write failing square-off tests.**

Cover:

```python
async def test_squareoff_is_noop_when_positions_are_flat():
    client = make_client(positions=[], quotes={})
    result = await square_off_nifty_positions(client, datetime(2026, 8, 7, 15, 0))
    assert result["success"] is True
    assert client.calls == []


async def test_squareoff_sells_a_long_nifty_position():
    client = make_client(
        positions=[{"exch": "NFO", "tsym": "NIFTY11AUG26C24450", "token": "41009", "netqty": "65"}],
        quotes={"41009": {"lp": 100.0}},
        remaining_positions=[],
    )
    result = await square_off_nifty_positions(client, datetime(2026, 8, 7, 15, 0))
    assert result["success"] is True
    assert client.calls[0]["side"] == "SELL"
    assert client.calls[0]["quantity"] == 65


async def test_squareoff_buys_back_a_short_nifty_position():
    client = make_client(
        positions=[{"exch": "NFO", "tsym": "NIFTY11AUG26C24450", "token": "41009", "netqty": "-65"}],
        quotes={"41009": {"lp": 100.0}},
        remaining_positions=[],
    )
    result = await square_off_nifty_positions(client, datetime(2026, 8, 7, 15, 0))
    assert result["success"] is True
    assert client.calls[0]["side"] == "BUY"


async def test_squareoff_refuses_unknown_non_nifty_exposure():
    client = make_client(
        positions=[{"exch": "NFO", "tsym": "BANKNIFTY11AUG26C50000", "token": "51009", "netqty": "15"}],
        quotes={"51009": {"lp": 100.0}},
    )
    result = await square_off_nifty_positions(client, datetime(2026, 8, 7, 15, 0))
    assert result["success"] is False
    assert result["reason"] == "UNKNOWN_REMOTE_EXPOSURE"
    assert client.calls == []


async def test_squareoff_fails_if_positions_remain_after_exit():
    client = make_client(
        positions=[{"exch": "NFO", "tsym": "NIFTY11AUG26C24450", "token": "41009", "netqty": "65"}],
        quotes={"41009": {"lp": 100.0}},
        remaining_positions=[{"exch": "NFO", "tsym": "NIFTY11AUG26C24450", "token": "41009", "netqty": "65"}],
    )
    result = await square_off_nifty_positions(client, datetime(2026, 8, 7, 15, 0))
    assert result["success"] is False
    assert result["reason"] == "POSITION_NOT_FLAT"
```

The module must provide `run_all()` that invokes each async test with `asyncio.run()` and never uses a live Flattrade token.

- [ ] **Step 2: Run the tests and confirm the command is missing.**

Run: `python -c "import test_squareoff; test_squareoff.test_squareoff_is_noop_when_positions_are_flat()"`

Expected: FAIL because `flattrade_bot.squareoff` is missing.

- [ ] **Step 3: Implement position filtering and opposite-side exits.**

The command must:

1. Load the session token only from `FLATTRADE_TOKEN` or the mode-0600 session-token path; if neither exists, perform a fresh automated login using the protected env file.
2. Call PositionBook and fail if the response is unavailable.
3. Treat `exch == "NFO"` and `tsym` beginning with `NIFTY` as bot-scope positions.
4. Refuse to touch any non-NIFTY nonzero position and emit a critical alert.
5. For a positive `netqty`, submit a SELL for `netqty`; for a negative `netqty`, submit a BUY for `abs(netqty)`.
6. Fetch a current quote for each token before submitting the aggressive-limit market substitute.
7. Confirm each exit through OrderBook and TradeBook.
8. Re-read PositionBook and require all bot-scope `netqty` values to be zero.
9. Redact all credentials and raw broker responses from both output and logs.

- [ ] **Step 4: Add a `--dry-run` mode.**

`python -m flattrade_bot.squareoff --dry-run` must print only redacted symbols, quantities, and intended sides, and must never call PlaceOrder. The systemd timer must not use `--dry-run`.

- [ ] **Step 5: Run the square-off tests and py_compile.**

Run: `python -c "import test_squareoff; test_squareoff.run_all(); print('SQUAREOFF_TESTS_PASS')"`

Run: `python -m py_compile flattrade_bot\squareoff.py flattrade_bot\execution.py flattrade_bot\broker\client.py`

Expected: `SQUAREOFF_TESTS_PASS` and no compiler output.

- [ ] **Step 6: Commit.**

```bash
git add flattrade_bot/squareoff.py flattrade_bot/execution.py flattrade_bot/broker/client.py test_squareoff.py README.md
git commit -m "Add independent broker position square-off"
```

### Task 5: Add Heartbeat, Redacted Rotating Logs, and Health Checks

**Files:**
- Create: `flattrade_bot/healthcheck.py`
- Create: `test_healthcheck.py`
- Modify: `flattrade_bot/main.py`
- Modify: `flattrade_bot/utils/discord.py`
- Modify: `flattrade_bot/broker/auto_login.py`
- Modify: `flattrade_bot/broker/client.py`

**Interfaces:**
- `write_heartbeat(path: Path, now: datetime) -> None` atomically writes an ISO-8601 timestamp with mode `0600`.
- `check_heartbeat(path: Path, now: datetime, max_age_seconds: int = 90) -> dict[str, Any]` returns `healthy`, `age_seconds`, and a redacted reason.
- CLI entry point: `python -m flattrade_bot.healthcheck` returns `0` when healthy and `1` when stale/missing.

- [ ] **Step 1: Write failing heartbeat tests.**

```python
def test_fresh_heartbeat_is_healthy(tmp_path):
    path = tmp_path / "heartbeat"
    now = datetime(2026, 8, 7, 9, 15, 0)
    write_heartbeat(path, now)
    result = check_heartbeat(path, now + timedelta(seconds=20))
    assert result["healthy"] is True


def test_stale_heartbeat_is_unhealthy(tmp_path):
    path = tmp_path / "heartbeat"
    now = datetime(2026, 8, 7, 9, 15, 0)
    write_heartbeat(path, now)
    result = check_heartbeat(path, now + timedelta(seconds=91))
    assert result["healthy"] is False
    assert result["reason"] == "HEARTBEAT_STALE"


def test_invalid_heartbeat_is_unhealthy(tmp_path):
    path = tmp_path / "heartbeat"
    path.write_text("not-a-time", encoding="utf-8")
    result = check_heartbeat(path, datetime(2026, 8, 7, 9, 15, 0))
    assert result["healthy"] is False
    assert result["reason"] == "HEARTBEAT_INVALID"
```

- [ ] **Step 2: Implement atomic heartbeat updates.**

Write to a mode-0600 temporary file in the same directory, flush and fsync it, and replace the target with `os.replace`. The main loop must update the heartbeat only after a successful market tick or after a handled error is recorded; a stuck network call must eventually make the heartbeat stale.

- [ ] **Step 3: Replace unbounded log growth with `RotatingFileHandler`.**

Use a 10 MiB maximum and seven backups. Add a single redaction filter that masks values matching `FLATTRADE_PASSWORD`, `FLATTRADE_API_KEY`, `FLATTRADE_API_SECRET`, `FLATTRADE_TOTP_KEY`, `FLATTRADE_TOKEN`, `request_code`, `jKey`, and `api_secret`. Never log full broker request or response bodies.

- [ ] **Step 4: Add the health CLI and test it.**

Run: `python -c "import test_healthcheck; test_healthcheck.run_all(); print('HEALTHCHECK_TESTS_PASS')"`

Expected: `HEALTHCHECK_TESTS_PASS`.

- [ ] **Step 5: Commit.**

```bash
git add flattrade_bot/healthcheck.py test_healthcheck.py flattrade_bot/main.py flattrade_bot/utils/discord.py flattrade_bot/broker/auto_login.py flattrade_bot/broker/client.py
git commit -m "Add bot heartbeat and redacted health monitoring"
```

### Task 6: Create Reproducible Runtime Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `requirements.lock`
- Create: `test_runtime_dependencies.py`
- Modify: `README.md`

**Interfaces:**
- `requirements.txt` contains exactly these direct runtime dependencies: `httpx`, `python-dotenv`, `pyotp`, `rich`, and `selenium`; `numpy` and `pandas` remain development/backtest dependencies unless a live import audit proves otherwise.
- `requirements.lock` contains the exact versions from a tested Python 3.11 environment; no local paths, editable installs, credentials, or Windows-only packages.

- [ ] **Step 1: Inventory runtime imports.**

Run: `python -c "import importlib; modules = ['httpx', 'dotenv', 'pyotp', 'rich', 'selenium']; [importlib.import_module(name) for name in modules]; print('RUNTIME_IMPORTS_PASS')"`

Expected: `RUNTIME_IMPORTS_PASS`.

- [ ] **Step 2: Add direct dependencies and generate the lock file from the validated environment.**

Use version ranges in `requirements.txt` that exclude known breaking major versions, install `pip-tools` in the development environment, and run `pip-compile --generate-hashes --output-file=requirements.lock requirements.txt`. Do not include the development-only Graphify, security CLI, or test tooling in the production runtime lock.

- [ ] **Step 3: Test clean installation in a temporary virtual environment.**

Run on Linux: `python3.11 -m venv /tmp/flattrade-deps-check && /tmp/flattrade-deps-check/bin/python -m pip install --require-hashes -r requirements.lock`

Expected: installation succeeds without using the repository `.env`.

- [ ] **Step 4: Scan dependencies.**

Run: `pip-audit -r requirements.txt`

Expected: no known high or critical vulnerabilities. Any unresolved finding blocks live deployment.

- [ ] **Step 5: Commit.**

```bash
git add requirements.txt requirements.lock test_runtime_dependencies.py README.md
git commit -m "Pin production runtime dependencies"
```

### Task 7: Add AWS Lightsail Bootstrap and Host Hardening

**Files:**
- Create: `deploy/aws-lightsail/bootstrap.sh`
- Create: `deploy/aws-lightsail/verify-host.sh`
- Create: `deploy/aws-lightsail/flattrade-bot.env.example`
- Create: `deploy/aws-lightsail/README.md`
- Create: `deploy/aws-lightsail/SECURITY.md`

**Interfaces:**
- `bootstrap.sh` is idempotent, contains no secrets, exits on errors, and installs only the packages required for Ubuntu, Python, headless Chrome/Selenium, time synchronization, firewalling, and unattended security updates.
- `verify-host.sh` exits nonzero unless timezone, IPv4 egress, firewall, SSH configuration, service user, secret file permissions, and required directories satisfy the deployment contract.
- `.env.example` documents variable names only; the real file is created manually under `/etc/flattrade-bot/flattrade.env` and is never copied into Git.

- [ ] **Step 1: Write host-verification tests as shell assertions.**

The verification script must check:

```text
OS is Ubuntu 24.04 LTS x86_64
timezone is Asia/Kolkata
curl -4 ifconfig.me returns a single IPv4 address
the IPv4 is not empty and is recorded for Flattrade allowlisting
the flattrade user exists and has no sudo rule
UFW default incoming policy is deny
SSH password authentication is disabled
SSH root login is disabled
/etc/flattrade-bot/flattrade.env is mode 0640 and owned by root:flattrade
/run/flattrade-bot is mode 0750 and owned by flattrade:flattrade
no process listens on HTTP, HTTPS, RDP, Jupyter, or arbitrary dashboard ports
```

- [ ] **Step 2: Implement idempotent bootstrap.**

Install Ubuntu packages with `apt-get`, enable `unattended-upgrades` and `chrony`/systemd time synchronization, create a separate `flattrade-admin` SSH administrator and a `flattrade` service user without sudo access, create `/opt/flattrade-bot`, `/etc/flattrade-bot`, and `/run/flattrade-bot`, and set ownership/modes explicitly. Do not create AWS access keys on the host.

- [ ] **Step 3: Install headless Chrome securely.**

Use the official signed Ubuntu/Google package source, verify the repository signing key fingerprint against the provider's current official documentation before enabling it, install the x86_64 browser, and run a non-trading Selenium smoke test. Do not download a random ChromeDriver binary or run a browser as root.

- [ ] **Step 4: Configure SSH hardening safely.**

Create an `/etc/ssh/sshd_config.d/90-flattrade-hardening.conf` snippet with:

```text
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
X11Forwarding no
AllowUsers flattrade-admin
```

Validate with `sshd -t` before restarting SSH and keep an already-open console session until a second key-authenticated session succeeds.

- [ ] **Step 5: Configure the provider and host firewalls.**

In Lightsail, allow TCP 22 only from the administrator's current public `/32`. On Ubuntu, run `ufw default deny incoming`, `ufw default allow outgoing`, and allow SSH only from that same `/32`. Do not open port 5000, because the automated login path captures its browser redirect without exposing a public OAuth listener.

- [ ] **Step 6: Verify the hardened host.**

Run: `sudo /opt/flattrade-bot/deploy/aws-lightsail/verify-host.sh`

Expected: `HOST_SECURITY_CHECK_PASS`.

- [ ] **Step 7: Commit.**

```bash
git add deploy/aws-lightsail/bootstrap.sh deploy/aws-lightsail/verify-host.sh deploy/aws-lightsail/flattrade-bot.env.example deploy/aws-lightsail/README.md deploy/aws-lightsail/SECURITY.md
git commit -m "Add AWS Lightsail host hardening"
```

### Task 8: Add systemd Service, Start, Square-Off, Health, and Stop Timers

**Files:**
- Create: `deploy/aws-lightsail/systemd/flattrade-bot.service`
- Create: `deploy/aws-lightsail/systemd/flattrade-bot-start.timer`
- Create: `deploy/aws-lightsail/systemd/flattrade-bot-squareoff.service`
- Create: `deploy/aws-lightsail/systemd/flattrade-bot-squareoff.timer`
- Create: `deploy/aws-lightsail/systemd/flattrade-bot-health.service`
- Create: `deploy/aws-lightsail/systemd/flattrade-bot-health.timer`
- Create: `deploy/aws-lightsail/systemd/flattrade-bot-stop.service`
- Create: `deploy/aws-lightsail/systemd/flattrade-bot-stop.timer`
- Test: `test_systemd_units.py`

**Interfaces:**
- `flattrade-bot.service` runs the main process under `flattrade`.
- `flattrade-bot-start.timer` starts it at `09:05:00 Asia/Kolkata` Monday through Friday.
- `flattrade-bot-squareoff.timer` runs the independent square-off at `15:00:00 Asia/Kolkata` Monday through Friday.
- `flattrade-bot-health.timer` runs the heartbeat check every minute during the day.
- `flattrade-bot-stop.timer` stops the main service at `15:05:00 Asia/Kolkata` Monday through Friday.

- [ ] **Step 1: Write unit-file validation tests.**

`test_systemd_units.py` must read every unit as text and assert:

```python
assert "User=flattrade" in main_service
assert "--auto-login --live-orders" in main_service
assert "OnCalendar=Mon..Fri *-*-* 09:05:00 Asia/Kolkata" in start_timer
assert "OnCalendar=Mon..Fri *-*-* 15:00:00 Asia/Kolkata" in squareoff_timer
assert "OnCalendar=Mon..Fri *-*-* 15:05:00 Asia/Kolkata" in stop_timer
assert "Persistent=false" in start_timer
assert "Persistent=false" in squareoff_timer
assert "Persistent=false" in stop_timer
assert "NoNewPrivileges=true" in main_service
assert "ProtectHome=true" in main_service
assert "PrivateTmp=true" in main_service
assert "Restart=on-failure" in main_service
```

- [ ] **Step 2: Create the main service with least privilege.**

The service must include `Type=exec`, `User=flattrade`, `Group=flattrade`, `WorkingDirectory=/opt/flattrade-bot`, an environment file, `ExecStart=/opt/flattrade-bot/.venv/bin/python -m flattrade_bot.main --auto-login --live-orders`, `Restart=on-failure`, `RestartSec=15s`, `TimeoutStartSec=10min`, `TimeoutStopSec=30s`, `KillMode=control-group`, `UMask=0077`, `NoNewPrivileges=true`, `PrivateTmp=true`, `ProtectHome=true`, `ProtectSystem=full`, `ProtectKernelTunables=true`, `ProtectKernelModules=true`, `ProtectControlGroups=true`, `RestrictSUIDSGID=true`, `RestrictRealtime=true`, `RestrictNamespaces=true`, `LockPersonality=true`, and `SystemCallArchitectures=native`.

Set `HOME`, `XDG_CONFIG_HOME`, and `XDG_CACHE_HOME` under `/run/flattrade-bot/home` so Chrome does not require writable access to a real user home directory. Permit writes only to `/run/flattrade-bot` and the rotating application log directory.

- [ ] **Step 3: Create the start timer.**

Use:

```ini
[Unit]
Description=Start Flattrade bot before the Indian market session

[Timer]
OnCalendar=Mon..Fri *-*-* 09:05:00 Asia/Kolkata
AccuracySec=1s
Persistent=false
Unit=flattrade-bot.service

[Install]
WantedBy=timers.target
```

- [ ] **Step 4: Create the independent square-off service and timer.**

The square-off service runs `python -m flattrade_bot.squareoff`, uses the same secure environment file and runtime token path, has `Type=oneshot`, `User=flattrade`, the same filesystem/network sandbox, and no automatic restart. Its timer uses `15:00:00 Asia/Kolkata`, `AccuracySec=1s`, and `Persistent=false`.

- [ ] **Step 5: Create the health service and timer.**

The health service runs `python -m flattrade_bot.healthcheck` as root only because it must restart a system service; the health module itself reads only the mode-0600 heartbeat and emits redacted status. If the heartbeat is stale, it sends a redacted Discord alert and requests a restart of `flattrade-bot.service`. The restart is safe only because Task 3 blocks new entries when remote exposure exists and Task 4 handles square-off independently.

- [ ] **Step 6: Create the stop service and timer.**

The stop service runs `systemctl stop flattrade-bot.service`; its timer fires at `15:05:00 Asia/Kolkata`. It must not place or cancel an exit order itself.

- [ ] **Step 7: Validate unit files on Linux.**

Run: `systemd-analyze verify deploy/aws-lightsail/systemd/*.service deploy/aws-lightsail/systemd/*.timer`

Run: `systemd-analyze calendar 'Mon..Fri *-*-* 09:05:00 Asia/Kolkata'`

Expected: no unit errors and the next trigger displayed in IST.

- [ ] **Step 8: Commit.**

```bash
git add deploy/aws-lightsail/systemd test_systemd_units.py
git commit -m "Schedule supervised trading lifecycle"
```

### Task 9: Add Safe Deployment and AWS Console Runbook

**Files:**
- Create: `deploy/aws-lightsail/deploy.sh`
- Create: `docs/AWS_LIGHTSAIL_DEPLOYMENT.md`
- Create: `docs/SECURITY_RUNBOOK.md`
- Create: `docs/OPERATIONS_RUNBOOK.md`
- Modify: `README.md`

**Interfaces:**
- `deploy.sh` accepts a user-provided server address through an environment variable or command argument, never through a committed config file, and refuses to run if `.env`, private keys, or untracked credential files are included in the rsync source.
- The deployment runbook describes AWS console actions, SSH key setup, static-IP verification, Flattrade allowlisting, dependency installation, systemd installation, smoke mode, rollback, and emergency square-off.

- [ ] **Step 1: Write deployment-script safety tests.**

The script must fail before upload if any of these are present in the source tree:

```text
.env
*.pem
id_rsa
id_ed25519
session.token
logs/bot.log
```

It must use an rsync exclude list for `.git`, `.env*`, `logs`, `graphify-out`, `__pycache__`, `.claude`, `.claude-flow`, and local runtime token files.

- [ ] **Step 2: Implement AWS console provisioning instructions.**

Document these exact selections:

```text
AWS Lightsail -> Create instance
Region: Mumbai / ap-south-1
Platform: Linux/Unix
Blueprint: Ubuntu 24.04 LTS x86_64
Bundle: 2 GB RAM / 2 vCPU / 60 GB SSD
Networking: attach a static public IPv4
Authentication: dedicated Ed25519 SSH key
```

Enable Lightsail account MFA, create a billing alarm, and do not create long-lived AWS access keys on the VPS.

- [ ] **Step 3: Install and configure the bot without secrets in the repository.**

Upload code, create `/opt/flattrade-bot/.venv`, install `requirements.lock`, create the service user, and create `/etc/flattrade-bot/flattrade.env` manually with mode `0640`, owner `root`, and group `flattrade`. The env file must contain `FLATTRADE_USER_ID`, `FLATTRADE_API_KEY`, `FLATTRADE_API_SECRET`, `FLATTRADE_TOTP_KEY`, `FLATTRADE_PASSWORD`, `DISCORD_WEBHOOK_URL`, `SESSION_START=09:15`, `SESSION_END=15:00`, and `TRADING_TIMEZONE=Asia/Kolkata`.

- [ ] **Step 4: Verify the actual public IPv4 before API allowlisting.**

Run on the server:

```bash
curl -4 https://ifconfig.me
python -c "import socket; print(socket.getaddrinfo('piconnect.flattrade.in', 443, socket.AF_INET))"
```

Record the returned public IPv4 in the private deployment record and update the Flattrade Wall API whitelist. Never place it in source code or a public issue.

- [ ] **Step 5: Install systemd units but start in read-only mode first.**

Install the service files, run `systemctl daemon-reload`, enable timers, and temporarily use a service command without `--live-orders` for the first deployment smoke. Confirm authentication, token storage, contract search, warmup, heartbeat, and Discord notifications before switching the service to live mode.

- [ ] **Step 6: Run the five-day deployment smoke test.**

For five trading days, verify:

```text
start timer fires at 09:05 IST
login completes after 05:00 IST
heartbeat updates at least every 90 seconds
no token/TOTP/password appears in journalctl or bot.log
no entry is submitted in read-only mode
the 15:00 square-off command reports no-op when flat
the 15:05 stop timer stops the service
```

- [ ] **Step 7: Perform one explicitly confirmed live lot.**

Only after the account is funded, the static IP is accepted, and the five-day smoke is clean, run a one-lot live test with explicit user confirmation. Verify PlaceOrder, OrderBook `COMPLETE`, TradeBook fill, PositionBook quantity, and the independent square-off path. A PlaceOrder `stat=Ok` without a fill is not success.

- [ ] **Step 8: Document rollback.**

The emergency procedure must be:

1. Run the independent square-off command from the server or use the Flattrade web terminal.
2. Stop `flattrade-bot.service` and all timers.
3. Revoke/rotate Flattrade API credentials if the host may be compromised.
4. Remove the VPS IPv4 from Flattrade Wall.
5. Preserve redacted `journalctl` and OrderBook/TradeBook evidence.
6. Rebuild the instance from a clean Ubuntu image instead of trusting a suspected compromised host.

- [ ] **Step 9: Commit.**

```bash
git add deploy/aws-lightsail/deploy.sh docs/AWS_LIGHTSAIL_DEPLOYMENT.md docs/SECURITY_RUNBOOK.md docs/OPERATIONS_RUNBOOK.md README.md
git commit -m "Document secure AWS Lightsail deployment"
```

### Task 10: Security and Release Gate

**Files:**
- Modify: `docs/SECURITY_RUNBOOK.md`
- Create: `test_release_gate.py`
- No production secrets or cloud account credentials are added.

**Interfaces:**
- `test_release_gate.py` is a local release checklist expressed as executable assertions for source-level requirements.
- The release gate returns nonzero if live order code can run without reconciliation, if an env file is tracked, if any systemd timer uses `Persistent=true`, or if the deployment includes a public debug port.

- [ ] **Step 1: Add source-level release assertions.**

Assert:

```python
from pathlib import Path
from subprocess import check_output

from dotenv import dotenv_values

tracked_files = [Path(item) for item in check_output(["git", "ls-files"], text=True).splitlines()]
tracked_source = "\n".join(
    path.read_text(encoding="utf-8")
    for path in tracked_files
    if path.suffix in {".py", ".md", ".sh", ".service", ".timer", ".txt"}
)
all_timer_contents = "\n".join(
    path.read_text(encoding="utf-8")
    for path in Path("deploy/aws-lightsail/systemd").glob("*.timer")
)
login_source = Path("flattrade_bot/broker/auto_login.py").read_text(encoding="utf-8")
secrets = [value for value in dotenv_values(".env").values() if value]

assert "reconcile_remote_state" in Path("flattrade_bot/main.py").read_text()
assert "square_off_nifty_positions" in Path("flattrade_bot/squareoff.py").read_text()
assert "--live-orders" in Path("deploy/aws-lightsail/systemd/flattrade-bot.service").read_text()
assert "Persistent=false" in all_timer_contents
assert "Persistent=true" not in all_timer_contents
assert not any(secret in tracked_source for secret in secrets)
assert "TOTP code:" not in login_source
```

- [ ] **Step 2: Run all local quality gates.**

Run:

```bash
python -m compileall -q flattrade_bot
python -m py_compile test_config.py test_session_store.py test_auto_login_security.py test_reconciliation.py test_squareoff.py test_healthcheck.py test_runtime_dependencies.py test_systemd_units.py test_release_gate.py
python -c "import test_config; test_config.run_all(); import test_session_store; test_session_store.run_all(); import test_reconciliation; test_reconciliation.run_all(); import test_squareoff; test_squareoff.run_all(); import test_healthcheck; test_healthcheck.run_all(); import test_systemd_units; test_systemd_units.run_all(); import test_runtime_dependencies; test_runtime_dependencies.run_all(); import test_release_gate; test_release_gate.run_all(); print('ALL_RELEASE_TESTS_PASS')"
npx @claude-flow/cli security scan --depth full
npx @claude-flow/cli security validate --check secrets
pip-audit -r requirements.txt
git diff --check
```

Expected: `ALL_RELEASE_TESTS_PASS`, no high/critical security findings, no secrets, no dependency findings requiring a block, and no whitespace errors in tracked source files.

- [ ] **Step 3: Run Linux host gates after deployment.**

Run:

```bash
sudo systemd-analyze security flattrade-bot.service
sudo systemctl list-timers --all 'flattrade-bot-*'
sudo ss -lntup
sudo ufw status verbose
sudo journalctl -u flattrade-bot.service --since today --no-pager
sudo /opt/flattrade-bot/deploy/aws-lightsail/verify-host.sh
```

Expected: only SSH is externally reachable, timers show IST schedules, the service runs as `flattrade`, and logs contain no credentials.

- [ ] **Step 4: Complete a paper-trading soak before live mode.**

Run the service without `--live-orders` for five market days. Compare signals against the local reference behavior, confirm strike rotation reset/re-warm, verify no duplicate process exists, and inspect memory usage during Selenium login and the two-second polling loop.

- [ ] **Step 5: Complete the one-lot live acceptance gate.**

The gate passes only when all of these are true:

```text
Flattrade accepts the AWS static IPv4
Daily login succeeds without exposing secrets
Remote state is flat before startup
One entry has OrderBook COMPLETE and TradeBook fill evidence
Local position appears only after fill confirmation
SL/target monitoring works
Independent square-off closes the position and verifies netqty=0
The 15:05 stop timer stops the main process
The health timer detects a deliberately stale heartbeat
The account contains no unrelated NFO positions
```

- [ ] **Step 6: Commit the release gate and final documentation.**

```bash
git add docs/SECURITY_RUNBOOK.md test_release_gate.py
git commit -m "Add secure live deployment release gate"
```

## Acceptance Criteria

- AWS Lightsail Mumbai instance has one persistent public IPv4 and that exact address is accepted by Flattrade.
- The VPS can be rebuilt without changing the documented secret-handling procedure.
- No runtime secret is present in Git, deployment archives, process arguments, logs, Discord messages, or systemd unit files.
- A normal SSH disconnect does not stop the bot; a systemd stop does stop the entire process group.
- The bot starts at 09:05 IST on valid weekdays and does not catch up after an outage.
- The risk gate permits entries only between configured 09:15 and 15:00 IST.
- Broker state is reconciled before any live entry and unknown remote exposure causes safe halt.
- A separate square-off command can close positions even when the main process's local state is empty.
- A stale heartbeat is detected and alerted without opening a replacement position blindly.
- The bot confirms fills through OrderBook/TradeBook before creating or clearing local position state.
- A five-day read-only soak and one-lot live test pass before normal live operation.
- Security scan, secret validation, dependency audit, unit-file validation, and host-hardening checks pass.

## Review Checklist

- [ ] Every changed Python file has a focused direct-function test.
- [ ] Every new systemd unit is validated with `systemd-analyze verify`.
- [ ] Every live-order path has a fail-closed response for broker API errors.
- [ ] Every secret-bearing path has a redaction or permission test.
- [ ] The account-scope rule prevents square-off from touching unrelated instruments.
- [ ] The deployment does not rely on an interactive terminal, RDP, or a browser window.
- [ ] The plan does not claim that a VPS can make the system unhackable or guarantee broker execution.
- [ ] Graphify and codebase-memory are refreshed after the implementation is complete.
