"""Network helpers for broker API requests."""

import socket

_original_getaddrinfo = socket.getaddrinfo


def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


class force_ipv4:
    """Ensures DNS resolution is pinned to IPv4.

    Idempotent and permanent: the socket.getaddrinfo patch is installed once
    and NEVER restored. Restoring on context exit was the source of a
    RecursionError — with several coroutines sharing the global patch (the
    live bot fires ~6 concurrent REST calls per second), contexts exited out
    of order, leaving stale wrappers chained on top of each other. After
    enough rounds the wrapper chain exceeded the interpreter recursion
    limit and every broker request died ("maximum recursion depth exceeded").

    The patch is process-wide and harmless to keep: Flattrade's PiConnect
    API only accepts IPv4 (the account is registered to an IPv4 wall).
    """

    def __enter__(self):
        _ensure_ipv4_patch()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _ensure_ipv4_patch() -> None:
    """Installs the IPv4-only getaddrinfo exactly once."""
    if socket.getaddrinfo is not ipv4_only_getaddrinfo:
        socket.getaddrinfo = ipv4_only_getaddrinfo