"""Miniature kit check module (governance corpus fixture; never executed)."""


def check_lock_08(ctx):
    """LOCK-08: literal integrity booleans in delivery code fail."""
    return ctx.scan_literals()


def check_lock_11(ctx):
    """LOCK-11: predictions on the primary and every guard."""
    return ctx.predictions()
