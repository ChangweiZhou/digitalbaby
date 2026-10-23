"""Stage 4: does a learner that generalised addition learn multiplication better, and forget addition less?

Development diagnostic only. Phase A trains addition exactly as the stage 1/2 runs of diag.py (same code path, so
its outcome is checked against the stored stage-1/2 run). Phase B then trains multiplication only, with a fresh
optimizer and the same regime for every arm, on the SAME split: the 80 taught pairs are taught as `a*b=`, the 20 HELD
pairs stay unseen in both operations. Addition is not rehearsed in phase B; its retention is measured throughout.

Arms (phase A):
  FRESH  no phase A (newborn net)
  MEM    addition, wd 0                      (memorised addition; ~6% unseen +)
  WD     addition, wd 1                      (~44% unseen +)
  GEN    addition + counting, wd 1           (~65% unseen +)
  SHUF   addition with answers permuted across the taught prompts + counting, wd 1 (same bytes and regime as GEN,
         no addition rule)
Phase B regime: full-batch AdamW lr 1e-3, weight decay 0 or 1 (both run from the same phase-A state).
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import diag

HERE = Path(__file__).resolve().parent
ARMS = {"FRESH": None, "MEM": (0.0, "none"), "WD": (1.0, "none"), "GEN": (1.0, "count"), "SHUF": (1.0, "count")}
B_LOG = [10, 20, 30, 40] + list(range(50, 1001, 50)) + list(range(1250, 20001, 250))


def render(pair, op):
    a, b = pair
    return list(f"{a}{op}{b}=".encode()), list(str(a + b if op == "+" else a * b).encode())


def fit(net, items, steps, wd, log, evals):
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    x, y = diag.batch(items)
    curve = [(0, {k: diag.accuracy(net, v)[0] for k, v in evals.items()})] if evals else []
    for step in range(1, steps + 1):
        loss = F.cross_entropy(net(x)[0].reshape(-1, 257), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if evals and step in log:
            curve.append((step, {k: diag.accuracy(net, v)[0] for k, v in evals.items()}))
    return curve


def first_step(curve, key, level):
    return next((s for s, acc in curve if acc[key] >= level), None)


def answers(net, items, pairs):
    return {f"{a},{b}": bytes(g).decode("latin1") for (a, b), g in zip(pairs, diag.generate(net, items))}


def run(world, seed, arm, b_steps, b_wds):
    sp = diag.split(world)
    taught_pairs = diag.ordered(sp["old"]) + diag.ordered(sp["new"])
    held_pairs = diag.ordered(sp["held"])
    sets = {f"{op_name}_{part}": [render(p, op) for p in pairs]
            for op_name, op in (("add", "+"), ("mul", "*")) for part, pairs in (("taught", taught_pairs),
                                                                                 ("held", held_pairs))}
    t0 = time.process_time()
    net = diag.Net(seed)
    phase_a = None
    if ARMS[arm] is not None:
        wd, aux = ARMS[arm]
        add = list(sets["add_taught"])
        if arm == "SHUF":
            ans = [a for _, a in add]
            random.Random(world * 10 + seed).shuffle(ans)
            add = [(p, a) for (p, _), a in zip(add, ans)]
        items = add + (diag.counting(world, False) if aux == "count" else [])
        fit(net, items, 20000, wd, (), {})
        phase_a = dict(wd=wd, aux=aux, add_taught=diag.accuracy(net, sets["add_taught"])[0],
                       add_held=diag.accuracy(net, sets["add_held"])[0], add_held_answers=answers(
                           net, sets["add_held"], held_pairs), probe_order=diag.probe(net, "digit"))
        if arm != "SHUF":
            # Same code path as the stored stage-1/2 diag.py run: the phase-A outcome must be identical.
            tag = "" if aux == "none" else f"_{aux}"
            ref = json.loads((HERE / "results" / "runs" /
                              f"batch_wd{wd:g}_digit_dec{tag}_{world}_s{seed}.json").read_text())
            phase_a["matches_stored_run"] = ({f"{h['pair'][0]},{h['pair'][1]}": h["got"] for h in ref["held"]}
                                             == phase_a["add_held_answers"])
    phase_b = {}
    for wd in b_wds:
        b = copy.deepcopy(net)
        curve = fit(b, sets["mul_taught"], b_steps, wd, set(B_LOG), sets)
        got = answers(b, sets["mul_held"], held_pairs)
        phase_b[f"wd{wd:g}"] = dict(
            curve=curve,
            steps_to_mul_taught_90=first_step(curve, "mul_taught", 0.9),
            steps_to_mul_taught_100=first_step(curve, "mul_taught", 1.0),
            mul_held_final=curve[-1][1]["mul_held"], mul_held_best=max(c[1]["mul_held"] for c in curve),
            add_taught_final=curve[-1][1]["add_taught"], add_held_final=curve[-1][1]["add_held"],
            mul_held_answers=got,
            mul_held_commutative_agreement=sum(got[f"{a},{b}"] == got[f"{b},{a}"] for a, b in held_pairs if a < b)
            / sum(1 for a, b in held_pairs if a < b),
            probe_order=diag.probe(b, "digit"))
    return dict(world=world, seed=seed, arm=arm, b_steps=b_steps, phase_a=phase_a, phase_b=phase_b,
                cpu_s=time.process_time() - t0, held_pairs=held_pairs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=tuple(ARMS), required=True)
    ap.add_argument("--b-steps", type=int, default=20000)
    ap.add_argument("--b-wd", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--save", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    r = run(a.world, a.seed, a.arm, a.b_steps, a.b_wd)
    with open(a.save, "w") as f:
        json.dump(r, f)
    brief = {k: {x: v[x] for x in ("steps_to_mul_taught_100", "mul_held_final", "add_taught_final", "add_held_final")}
             for k, v in r["phase_b"].items()}
    print(json.dumps(dict(arm=r["arm"], world=r["world"], seed=r["seed"], cpu_s=round(r["cpu_s"]),
                          phase_a=None if r["phase_a"] is None else {k: r["phase_a"][k] for k in r["phase_a"]
                                                                      if k != "add_held_answers"},
                          phase_b=brief)))
