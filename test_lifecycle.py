from flattrade_bot.utils.lifecycle import TerminalLifecycle


def test_terminal_lifecycle_requests_shutdown_only_once():
    calls = []
    lifecycle = TerminalLifecycle(lambda: calls.append("shutdown"), parent_pid=0)

    lifecycle.request_shutdown("test")
    lifecycle.request_shutdown("duplicate")

    assert calls == ["shutdown"]


def test_terminal_lifecycle_stops_when_parent_exits():
    import time
    import flattrade_bot.utils.lifecycle as lifecycle_module

    calls = []
    original_process_check = lifecycle_module._process_is_alive
    lifecycle_module._process_is_alive = lambda _pid: False
    lifecycle = TerminalLifecycle(lambda: calls.append("shutdown"), parent_pid=12345, poll_interval=0.01)
    try:
        lifecycle.start()
        for _ in range(20):
            if calls:
                break
            time.sleep(0.01)
    finally:
        lifecycle.stop()
        lifecycle_module._process_is_alive = original_process_check

    assert calls == ["shutdown"]
