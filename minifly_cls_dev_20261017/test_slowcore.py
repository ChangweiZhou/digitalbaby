"""Gradient checks for slowcore: BPTT against finite differences; e-prop equals BPTT when W_rec = 0 (then the
truncation e-prop makes is exact); e-prop with recurrence is an approximation and is only checked for alignment."""
import numpy as np

import slowcore


def batch():
    recs = [(b"3+8=", b"11"), (b"2+2=", b"4"), (b"9+0=", b"9")]
    return slowcore.sequences(recs, ticks=2)


def numeric(core, tok, tgt, key, idx, h=1e-6):
    w = core.p[key]
    old = w[idx]
    w[idx] = old + h
    lp = core.gradients(tok, tgt)[0]
    w[idx] = old - h
    lm = core.gradients(tok, tgt)[0]
    w[idx] = old
    return (lp - lm) / (2 * h)


def test_bptt_matches_finite_differences():
    core = slowcore.SlowCore(1, width=8, ticks=2, rule="bptt")
    rng = np.random.default_rng(0)
    core.p["W_out"] = rng.normal(0, 0.5, core.p["W_out"].shape)
    tok, tgt = batch()
    _, g = core.gradients(tok, tgt)
    for key in ("W_in", "W_rec", "b", "W_out"):
        flat = list(np.ndindex(core.p[key].shape))
        picks = [flat[i] for i in rng.choice(len(flat), 6, replace=False)]
        if key == "W_in":
            picks = [(j, int(t)) for j, t in zip(rng.integers(0, 8, 6), tok[0, :6])]
        for idx in picks:
            num = numeric(core, tok, tgt, key, idx)
            assert abs(num - g[key][idx]) < 1e-5 * max(1, abs(num)), (key, idx, num, g[key][idx])


def test_eprop_equals_bptt_without_recurrence():
    rng = np.random.default_rng(1)
    a = slowcore.SlowCore(2, width=8, ticks=2, rule="bptt")
    b = slowcore.SlowCore(2, width=8, ticks=2, rule="eprop", feedback="symmetric")
    for core in (a, b):
        core.p["W_rec"][:] = 0
        core.p["W_out"] = rng.normal(0, 0.5, core.p["W_out"].shape) if core is a else a.p["W_out"].copy()
    tok, tgt = batch()
    la, ga = a.gradients(tok, tgt)
    lb, gb = b.gradients(tok, tgt)
    assert abs(la - lb) < 1e-12
    for key in ("W_in", "b", "W_out", "c"):
        assert np.allclose(ga[key], gb[key], atol=1e-10), key


def test_eprop_aligned_with_bptt_with_recurrence():
    rng = np.random.default_rng(2)
    a = slowcore.SlowCore(3, width=16, ticks=2, rule="bptt")
    b = slowcore.SlowCore(3, width=16, ticks=2, rule="eprop")
    a.p["W_out"] = rng.normal(0, 0.5, a.p["W_out"].shape)
    b.p["W_out"] = a.p["W_out"].copy()
    tok, tgt = batch()
    ga, gb = a.gradients(tok, tgt)[1], b.gradients(tok, tgt)[1]
    for key in ("W_in", "W_rec"):
        cos = float((ga[key] * gb[key]).sum() / np.linalg.norm(ga[key]) / np.linalg.norm(gb[key]))
        assert cos > 0.3, (key, cos)


def test_learns_a_tiny_table():
    recs = [(b"1+1=", b"2"), (b"1+2=", b"3"), (b"2+1=", b"3"), (b"2+2=", b"4")]
    for rule in ("bptt", "eprop"):
        core = slowcore.SlowCore(4, width=32, ticks=1, rule=rule, lr=3e-3)
        for _ in range(600):
            core.step(recs)
        assert core.accuracy(recs) == 1.0, rule


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name, flush=True)
