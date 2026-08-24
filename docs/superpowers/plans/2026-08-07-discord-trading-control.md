# Discord Trading Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add authenticated Discord slash-command and weekday schedule control for the local Flattrade live trading process.

**Architecture:** A long-running local controller connects outbound to Discord's Gateway, registers guild slash commands, and owns the Flattrade child process. The trading engine handles a local stop request asynchronously, closes any open position through the broker, sends the existing Discord close notification, and exits only after the close is confirmed.

**Tech Stack:** Python 3.11, `discord.py`, `httpx`, `zoneinfo`, Windows Task Scheduler, existing Flattrade broker and webhook notifier.

## Global Constraints

- The Discord bot token and trading credentials stay in the ignored local `.env` file.
- Control commands require configured user, guild, and channel allowlists.
- The controller uses Discord Gateway slash commands and no public HTTP endpoint.
- Live startup uses `--auto-login --live-orders`.
- The weekday schedule uses `Asia/Kolkata`, `09:15`, and `15:00`.
- The controller never force-kills a process with an unconfirmed open position.
- The existing `ammu` webhook URL is not copied into source or committed.

---

### Task 1: Add failing control and shutdown tests

**Files:**
- Create: `test_control.py`
- Modify: `test_execution.py`
- Modify: `test_lifecycle.py`

**Steps:**

- [ ] Add tests for ID-list parsing, allowlist authorization, process state transitions, and session end detection.
- [ ] Add a test that a forced close submits a broker SELL and sends a close notification.
- [ ] Update the shutdown expectation to distinguish a shutdown request from immediate process termination.
- [ ] Run the focused tests and confirm they fail because the new interfaces do not exist.

### Task 2: Implement control configuration and process manager

**Files:**
- Modify: `flattrade_bot/config.py`
- Create: `flattrade_bot/control.py`
- Modify: `.env.example`

**Steps:**

- [ ] Add Discord token, guild, channel, user allowlist, timezone, start/stop time, and stop-file settings.
- [ ] Implement fail-closed ID parsing and authorization.
- [ ] Implement a process manager that starts `python -m flattrade_bot.main --auto-login --live-orders`, writes the local stop request, and reports status without force-killing the child.
- [ ] Run the focused control tests and confirm they pass.

### Task 3: Implement graceful EOD and remote stop

**Files:**
- Modify: `flattrade_bot/risk/manager.py`
- Modify: `flattrade_bot/execution.py`
- Modify: `flattrade_bot/main.py`

**Steps:**

- [ ] Add a session-complete predicate based on the configured end minute.
- [ ] Refactor confirmed exit handling into a reusable close method supporting `EOD` and manual stop reasons.
- [ ] Make the engine poll the stop request and close an active position before ending.
- [ ] Make the live loop stop after the 15:00 exit path completes.
- [ ] Run execution, lifecycle, and focused shutdown tests.

### Task 4: Add the Discord Gateway controller and schedule

**Files:**
- Create: `flattrade_bot/discord_control.py`
- Create: `START_DISCORD_CONTROL.bat`

**Steps:**

- [ ] Create a guild-scoped slash-command group with `start`, `stop`, and `status` actions.
- [ ] Enforce the configured allowlist before any process action.
- [ ] Add the weekday `09:15`/`15:00` Asia/Kolkata scheduler.
- [ ] Add actionable validation errors for missing bot token or IDs.
- [ ] Add a Windows launcher with the project working directory.

### Task 5: Verify and document activation

**Files:**
- Modify: `README.md`

**Steps:**

- [ ] Install `discord.py` into the active Python environment.
- [ ] Run compile checks and the full existing test suite.
- [ ] Validate the current webhook without printing its URL; report the current Flattrade webhook status.
- [ ] Run the controller configuration check without connecting when credentials are absent.
- [ ] Document the Discord Developer Portal setup, local `.env` keys, minimum bot permissions, and Task Scheduler launch command.
- [ ] Test `/trading status`, `/trading start`, and `/trading stop` with the user's configured bot, without placing a live order.
