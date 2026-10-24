"""Unit checks for LEARNED-BANK V1 (cycle 1, extended in cycle 3 for PAR).
Run: python checks.py [out.json]  (prints one line per check; exits 1 on failure)."""
import json
import sys
import time

import numpy as np

import data
import lb1

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ("" if not detail else "  " + str(detail)), flush=True)


def jaccard(x, y):
    u = np.sum((x + y) > 0)
    return float(np.sum(x * y) / u) if u else 1.0


t0 = time.process_time()
world = data.make_world("D1")
all_pairs = [(a, b) for a in range(10) for b in range(10)]

# data checks
held, train = set(world["held_pairs"]), set(world["train_pairs"])
check("split disjoint and complete", not (held & train) and len(held | train) == 100,
      f"train {len(train)} held {len(held)}")
sched = data.schedule(world, 3)
check("schedule never contains held pairs", all(p not in held for p in sched))
der = data.derangement("MOD", world["train_pairs"], "D1")
check("derangement has no fixed point",
      all(der[p] != data.answer("MOD", *p) for p in world["train_pairs"]))
check("PAR answers are '0'/'1' for (a+b) mod 2",
      all(data.answer("PAR", a, b) == ord("01"[(a + b) % 2]) for a, b in all_pairs))
for wn in ("D1", "D2", "D3"):
    w = data.make_world(wn)
    sh = data.shuffled_labels("PAR", w)
    agree = sum(sh[p] == data.answer("PAR", *p) for p in w["train_pairs"])
    mirror = all(sh[(a, b)] == sh[(b, a)] for a, b in w["train_pairs"])
    check(f"{wn}: PAR SHUF labels cover the training pairs, orbit-consistent, agreement near half",
          set(sh) == set(w["train_pairs"]) and mirror and abs(agree - len(sh) / 2) <= 2,
          {"agree": agree, "n": len(sh)})
    check(f"{wn}: MOD SHUF is still the derangement",
          data.shuffled_labels("MOD", w) == data.derangement("MOD", w["train_pairs"], wn))

for arm in ("BASE", "FROZEN", "LEARN", "ORACLE"):
    m = lb1.Model(arm, world["model_seed"], eta_ramp=1)
    hashes = {c.fly.m.fast.tobytes() + c.fly.m.slow.tobytes() + c.fly.m.adapt.tobytes() for c in m.S.comp}
    check(f"{arm}: shared-bank compartments identical at birth", len(hashes) == 1)
    h0 = m.state_hash()
    pr = [m.probe(data.prompt(a, b), data.answer("MOD", a, b)) for a, b in [(3, 5), (0, 9), (7, 7)]]
    check(f"{arm}: probe leaves state unchanged", m.state_hash() == h0)
    zero = all(np.all(r["mS"] == 0) and (r["mL"] is None or np.all(r["mL"] == 0)) for r in pr)
    check(f"{arm}: opponent margins exactly zero at birth", zero)
    # code consistency between probe flow and trace_for
    same = all(np.array_equal(r["xS"], m.codes_for(data.prompt(a, b), data.answer("MOD", a, b))[0])
               for r, (a, b) in zip(pr, [(3, 5), (0, 9), (7, 7)]))
    if m.L is not None:
        same = same and all(np.array_equal(r["xL"], m.codes_for(data.prompt(a, b), data.answer("MOD", a, b))[1])
                            for r, (a, b) in zip(pr, [(3, 5), (0, 9), (7, 7)]))
    check(f"{arm}: probe codes equal trace_for codes", same)
    # identity gate (ORACLE codes depend on answer class by construction, so only S is gated there)
    codes = [m.codes_for(data.prompt(a, b), data.answer("MOD", a, b)) for a, b in all_pairs]
    distinct_S = len({c[0].tobytes() for c in codes}) == 100
    detail = {"S_distinct": len({c[0].tobytes() for c in codes})}
    ok = distinct_S
    if m.L is not None and arm != "ORACLE":
        nL = len({c[1].tobytes() for c in codes})
        detail["L_distinct"] = nL
        ok = ok and nL == 100
        offd = [jaccard(codes[i][1], codes[j][1]) for i in range(100) for j in range(i)]
        detail["L_mean_offdiag_jaccard"] = round(float(np.mean(offd)), 4)
    check(f"{arm}: identity gate (100 distinct prompt codes)", ok, detail)
    again = m.codes_for(data.prompt(4, 2), data.answer("MOD", 4, 2))
    first = m.codes_for(data.prompt(4, 2), data.answer("MOD", 4, 2))
    check(f"{arm}: same prompt twice gives identical code",
          all(np.array_equal(u, v) for u, v in zip(again, first) if u is not None))
    # teaching direction: one record with target '8' for 3+5
    target = data.answer("MOD", 3, 5)
    w_before = None if m.layer is None else m.layer.W1.copy()
    m.train(data.prompt(3, 5), target, target)
    m.train(data.prompt(3, 5), target, target)     # second record: eta ramp of 1 makes the layer learn here
    r = m.probe(data.prompt(3, 5), target)
    y = lb1.bits_of(target)
    okS = bool(np.all(np.sign(r["mS"]) == y))
    okL = True if r["mL"] is None else bool(np.all(np.sign(r["mL"]) == y))
    check(f"{arm}: two teaching records move every bit margin toward its target", okS and okL,
          {"S_byte": r["S"], "L_byte": r["L"], "target": target})
    if m.layer is not None:
        moved = not np.array_equal(w_before, m.layer.W1)
        check(f"{arm}: W1 {'changes' if arm == 'LEARN' else 'stays fixed'}", moved == (arm == "LEARN"))

# no-write equivalence: zero coefficients are exactly a passive event
b1, b2 = lb1.Bank(), lb1.Bank()
m0 = lb1.Model("BASE", 1)
x = m0.code_S(lb1.trace_for(data.prompt(2, 6)))
b1.passive(lb1.DELTA, x, lb1.DELTA)
b2.teach(x, np.zeros(lb1.NBITS), lb1.DELTA)
check("zero teaching coefficient equals passive event", b1.state_hash() == b2.state_hash())

cpu = time.process_time() - t0
ok = all(r["pass"] for r in results)
json.dump({"pass": ok, "cpu_s": round(cpu, 2), "checks": results}, open(sys.argv[1] if len(sys.argv) > 1 else
                                                                       "cycles/CHECKS.json", "w"), indent=1)
print("ALL PASS" if ok else "SOME FAILED", f"cpu {cpu:.1f}s")
sys.exit(0 if ok else 1)
