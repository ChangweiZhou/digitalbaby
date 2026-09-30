"""Unit tests for Package A V3-CLAUDE mechanisms (no science worlds, no calibration outcomes)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paths  # noqa: E402,F401
import stores  # noqa: E402
import systems  # noqa: E402
from fixture import DT  # noqa: E402

CUE = b"        1+2="


def _feed(m, cue=CUE, t0=0.0, answer=ord("3")):
    for i, b in enumerate(cue):
        m.byte(b, t0 + i * DT, learn=False)
    t = t0 + len(cue) * DT
    m.byte(answer, t, learn=False)
    return t


def _state(m):
    return m.fly.m.fast.copy(), m.fly.m.slow.copy(), m.fly.m.adapt.copy()


def test_separate_births_are_canonical_and_distinct():
    a, ra = stores.birth("native")
    b, rb = stores.birth("content")
    assert ra["canonical_B_sha256"] == rb["canonical_B_sha256"] == stores.EXPECTED_B
    assert a.fly is not b.fly and a.fly.m is not b.fly.m and a.fly.m.fast is not b.fly.m.fast


def test_signed_interface_matches_native_at_binary_targets_and_zero_writes_nothing():
    for r in (0, 1):
        nat, _ = stores.birth("native")
        sig, _ = stores.birth("native")
        t = _feed(nat); _feed(sig)
        nat.teach_logged(r, t, write=True)
        sig.teach_signed(1.0, float(r), t, write=True)
        for x, y in zip(_state(nat), _state(sig)):
            assert np.max(np.abs(x - y)) <= 1e-12
    off, _ = stores.birth("native")
    zero, _ = stores.birth("native")
    t = _feed(off); _feed(zero)
    off.teach_logged(1, t, write=False)
    assert zero.teach_signed(0.0, 0.0, t, write=True) == 0.0
    for x, y in zip(_state(off), _state(zero)):
        assert np.array_equal(x, y)


def test_signed_residual_sign_reversal():
    base, _ = stores.birth("native")
    pos, _ = stores.birth("native")
    neg, _ = stores.birth("native")
    t = _feed(base); _feed(pos); _feed(neg)
    base.teach_logged(0, t, write=False)
    a_pos = pos.teach_signed(0.0, 0.4, t, write=True)
    a_neg = neg.teach_signed(0.0, -0.4, t, write=True)
    assert a_pos > 0 and abs(a_pos - a_neg) <= 1e-12 * max(1.0, a_pos)
    for x0, xp, xn in zip(_state(base)[:2], _state(pos)[:2], _state(neg)[:2]):
        assert np.max(np.abs((xp - x0) + (xn - x0))) <= 1e-12
        assert np.max(np.abs(xp - x0)) > 0


def test_z0_resource_equals_native_and_z2_gate_formula():
    nat, _ = stores.birth("native")
    z0, _ = stores.birth_z("Z0_resource", 190000, 0)
    t = _feed(nat); _feed(z0)
    a1 = nat.teach_logged(1, t, write=True)
    a2 = z0.teach_logged(1, t, write=True)
    assert a1 == a2
    for x, y in zip(_state(nat), _state(z0)):
        assert np.array_equal(x, y)
    assert (z0.z_load[z0.z_mask] >= 0).all() and z0.z_load.max() > 0 and z0.z_load[~z0.z_mask].max() == 0
    z2, _ = stores.birth_z("Z2", 190000, 0)
    z2.z_load[z2.z_indices] = 0.5
    t = _feed(z2)
    z2.teach_logged(1, t, write=True)
    info = z2.last_z
    assert info["gated_slow_l1"] < info["native_slow_l1"]
    assert abs(info["gated_slow_l1"] - info["gate_target_l1"]) <= 1e-9 * max(1.0, info["native_slow_l1"])


def test_z2_rand_matches_bucket_l1():
    rng = np.random.default_rng(1)
    g = rng.uniform(0, 1, 400); g[::7] = 0.0
    slow = rng.normal(size=400); u = slow.copy(); side = rng.integers(0, 2, 400)
    out, buckets = stores._z2rand_gate(g, slow, u, side, "k")
    assert ((out >= 0) & (out <= 1)).all() and not np.array_equal(out, g)
    for _, (n, target, actual) in buckets.items():
        assert abs(target - actual) <= 1e-9 * max(1.0, target)


def test_p_support_budget_and_mechanisms():
    arms = {a: stores.birth_p(a)[0] for a in stores.P_ARMS}
    support = arms["P0"].p_support
    budget = arms["P0"].p_budget.copy()
    b0 = arms["P0"].fly.m.B.data.copy()
    for a, m in arms.items():
        for k in range(3):
            t = _feed(m, t0=k * 165.0, answer=ord("3"))
            m.teach_logged(1, t, write=True)
        B = m.fly.m.B
        assert m.p_support == support and m.p_support[2] == B.shape
        sums = np.bincount(B.indices, weights=B.data, minlength=B.shape[1])
        assert np.allclose(sums, budget, rtol=1e-12, atol=1e-15)
        assert np.isfinite(B.data).all() and (B.data > 0).all()
        if a == "P0":
            assert np.array_equal(B.data, b0)
        else:
            assert not np.array_equal(B.data, b0)
    # P2's stabilising term: at an active KC, larger relative weight decays (a_i - v) vs P1's +a_i
    rows = np.array([0, 1]); v = np.array([0.5, 3.0]); a = np.array([1.0, 1.0])
    d1 = stores.P_EPS * a[rows]
    d2 = stores.P_EPS * (a[rows] - v)
    assert (d1 > 0).all() and d2[0] > 0 > d2[1]


def test_p4_first_update_zero_and_order_reversal():
    def run(order):
        m, _ = stores.birth_p("P4")
        a = [np.zeros(88), np.zeros(88)]
        a[0][:10] = 1.0; a[1][40:60] = 1.0
        x = [np.zeros(m.n_native_kc), np.zeros(m.n_native_kc)]
        x[0][:300] = 1.0; x[1][2000:2300] = 1.0
        d0 = m.fly.m.B.data.copy()
        logs = []
        for step, i in enumerate(order):
            m._p_update(a[i][m.fly.m.pn_type_index], x[i], step * 100.0)
            logs.append(m.p_log[-1][0])
        return logs, m
    l_ab, m_ab = run([0, 1])
    l_ba, m_ba = run([1, 0])
    assert l_ab[0] == 0.0 and l_ba[0] == 0.0 and l_ab[1] > 0 and l_ba[1] > 0
    # an edge whose PN is active only in event 0 and whose KC is active only in event 1:
    # 0 -> 1 (pre before post) potentiates it, 1 -> 0 (post before pre) depresses it (before budget rescale)
    B = m_ab.fly.m.B
    pn_on = np.isin(m_ab.fly.m.pn_type_index[m_ab.p_rows], np.arange(10))
    kc_on = (m_ab.p_cols >= 2000) & (m_ab.p_cols < 2300)
    edges = np.flatnonzero(pn_on & kc_on)
    assert len(edges) > 0
    birth = stores.birth_p("P4")[0].fly.m.B.data
    rel_ab = m_ab.fly.m.B.data[edges] / birth[edges]
    rel_ba = m_ba.fly.m.B.data[edges] / birth[edges]
    assert np.median(rel_ab) > np.median(rel_ba)


def test_derangement_and_novelty():
    for key in ("a", "b", "c"):
        for n in (2, 4):
            p = systems.derangement(n, key)
            assert sorted(p) == list(range(n)) and all(p[i] != i for i in range(n))
    x = np.zeros(20, bool); x[:5] = True
    assert systems._jaccard_novelty(x, []) == 1.0
    assert systems._jaccard_novelty(x, [x.copy()]) == 0.0


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        print("PASS", f.__name__, flush=True)
    print(f"{len(fns)} tests passed")
