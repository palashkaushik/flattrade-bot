# Discord Trading Control Design

## Goal

Control the local Flattrade trading process from an authenticated Discord slash-command bot without exposing a public HTTP endpoint or accepting arbitrary remote commands.

## Scope

- Add a local Discord Gateway client using `discord.py`.
- Register guild-scoped `/trading start`, `/trading stop`, and `/trading status` commands.
- Accept commands only from configured Discord user IDs, guild ID, and channel ID.
- Start the current Flattrade process with `--auto-login --live-orders`.
- Stop the process through a local request file so the engine can close an open position before exiting.
- Run the weekday schedule in India Standard Time: start at `09:15`, stop at `15:00`.
- Preserve the existing webhook notifier for outbound trade alerts.

## Security

- The Discord bot token remains in the ignored local `.env` file.
- An empty allowlist rejects every control command.
- The control bot requests no message-content privileged intent.
- The controller does not expose a listening HTTP port.
- Stop requests are graceful; the controller does not force-kill a live trading process after a timeout.

## Failure Behavior

- Missing Discord configuration prevents the controller from starting.
- Duplicate start requests return the existing process status.
- Stop requests remain pending if an exit quote or broker close cannot be confirmed.
- Discord reconnects through the library's Gateway handling, but a disconnected controller does not restart a child with unknown broker exposure.
- A Discord alert webhook failure is logged and does not crash the trading engine.

## Testing

- Unit-test ID parsing and authorization.
- Unit-test process start/stop state transitions with a fake process.
- Unit-test the 15:00 session-complete behavior.
- Unit-test forced position close and Discord notification calls.
- Run the existing execution and lifecycle tests.
- Perform live Discord command testing only after the user enters the bot token and IDs locally; no test order will be placed.
