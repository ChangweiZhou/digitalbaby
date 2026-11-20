#!/usr/bin/env python3
"""Checks of replay4.py.  (1) The relay system's compiled loop against its numpy loop with dreams being made, counted and replayed, both phases.
(2) With q = 0 the program is arm R of replay.py: the same strengths after the same life (chunked calls of the same loop).  (3) The relay system's
writing of an answer agrees with its scoring.  (4) Dreams draw pairs that were never shown.      python3 check_replay4.py"""
import random
import numpy as np
import replay, replay4
NAMES = ("W1", "b1", "W2", "b2", "R")
def diff(a, b): return max(np.abs(getattr(a, k) - getattr(b, k)).max() / (np.abs(getattr(a, k)).max() + 1e-300) for k in NAMES)
if __name__ == "__main__":
    bad = 0; base = dict(seed="3", plan="add:600,sub:300", every="100", chunk="50", label="chk", nowrite="1")
    for kw in (dict(q="0"), dict(q="0.3"), dict(q="0.3", w="1/4"), dict(q="0.3", start="300"), dict(q="0.3", ans="chance")):
        res = []
        for fast in "01": o = replay4.run(dict(base, **kw, fast=fast)); res.append((replay4.run.last, o))
        (a, oa), (b, ob) = res; w = diff(a, b); same = [x["dreams_kept"] for x in oa["rows"]] == [x["dreams_kept"] for x in ob["rows"]]; bad += not (w < 1e-6 and same)
        print(f"{' '.join(f'{k}={v}' for k, v in kw.items()):20s}: largest difference over largest strength {w:.1e}; dreams {sum(x['dreams'] for x in oa['rows'])}, kept {sum(x['dreams_kept'] for x in oa['rows'])}, "
              f"of addition right {sum(x['dreams_add_right'] for x in oa['rows'])} of {sum(x['dreams_add'] for x in oa['rows'])}, on never-shown pairs {sum(x['dreams_add_never_shown'] for x in oa['rows'])}; wrong replays {sum(x['replays_wrong'] for x in oa['rows'])} of {sum(x['replays'] for x in oa['rows'])}")
    replay4.run(dict(base, q="0", fast="0")); a = replay4.run.last; replay.run(dict(seed="3", plan="add:600,sub:300", every="100", who="R", r="1", label="chk", nowrite="1", fast="0")); b = replay.run.last; w = diff(a, b); bad += not w < 1e-6
    print(f"q = 0 against arm R of replay.py (numpy, the same life): {w:.1e}")
    replay4.run(dict(seed="3", plan="add:30000", every="30000", chunk="500", q="0", label="chk", nowrite="1", fast="1")); a = replay4.run.last
    rng = np.random.default_rng(1); pairs = [(int(a), int(b)) for a, b in rng.integers(100, size=(300, 2))]; Ls = np.array([replay.text("+", a, b) for a, b in pairs]); Wr = replay4.write(a, Ls[:, :6])
    agree = float(100 * (Wr[:, 6:9] == Ls[:, 6:9]).all(1).mean()); sc = a.score("+", pairs); bad += abs(agree - sc) > 1e-9; print(f"after 30,000 lines, whole answers the relay system writes right against whole answers scored right, 300 pairs: {agree:.1f} and {sc:.1f}")
    c = replay4.Cells(8); L1, L2 = replay.text("+", 12, 34), replay.text("+", 56, 7); c.count(L1); c.count(L1[:8] + [9, replay.END], 0.5); r1 = c.answer(L1[:6]); r2 = c.answer(L2[:6]); bad += not (r1 is not None and r1[6:] == L1[6:] and r2 is None)
    print(f"the store's own answer: to a counted line (with a lighter wrong line beside it) {'right' if r1 and r1[6:] == L1[6:] else 'WRONG'}; to a line not counted: {r2}")
    o = replay4.run(dict(seed="3", plan="add:20000,sub:5000", every="5000", chunk="500", q="0.3", label="chk", nowrite="1", fast="1")); ok = True
    for ph in (0, 1):
        for k, v in o[f"whole_after_phase_{ph}"].items():
            ok &= v["from_store"] + v["from_relay"] + v["mixed"] == v["n"] and v["disagree_store_right"] + v["disagree_relay_right"] + v["disagree_neither"] == v["disagree"]
        h = o[f"whole_after_phase_{ph}"]["add"]; g = o[f"store_after_phase_{ph}"]["held_add"]; ok &= abs(h["relay_alone"] - g["relay"]) < 1e-9 and abs(h["store_alone"] - 100 * g["store_right"] / 500) < 1e-9
        print(f"the whole answer after phase {ph}, 500 held additions: whole {h['whole']:.1f}, relay alone {h['relay_alone']:.1f}, store alone {h['store_alone']:.1f}; from the store {h['from_store']}, from the relay system {h['from_relay']}, mixed {h['mixed']}; relay alone and store alone agree with the earlier measures: {ok}")
    q0 = replay4.run(dict(seed="3", plan="add:20000", every="20000", chunk="500", q="0", label="chk", nowrite="1", fast="1"))["whole_after_phase_0"]["add"]; ok &= q0["from_store"] == 0 and q0["whole"] == q0["relay_alone"]; bad += not ok
    print(f"with no dreams the whole answer on never-shown additions is the relay system's: {q0['whole']:.1f} and {q0['relay_alone']:.1f}, from the store {q0['from_store']}")
    print("CHECK", "FAILED" if bad else "PASSED")
