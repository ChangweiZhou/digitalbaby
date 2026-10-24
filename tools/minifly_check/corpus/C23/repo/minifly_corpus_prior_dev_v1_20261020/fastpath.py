"""Corpus stub fast path (never executed)."""


def step(loss):
    loss.backward()
    return loss
