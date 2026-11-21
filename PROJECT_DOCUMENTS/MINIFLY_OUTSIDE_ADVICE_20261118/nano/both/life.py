#!/usr/bin/env python3
"""nano/both, PLAN_3.md: the joined system of both.py through a life of two tasks, addition then subtraction (cells2.py's texts), with the route that
nano/replay found to keep the relay system: after each real line the relay system learns one line drawn from a store of counting cells
(nano/replay/replay.py's class Counting, K = 8, copied unchanged; drawn exactly as replay6.py's arm R draws). The program shows no addition in the
second phase. The cells that count of nano/merge and the two answering cells read the real lines only.
    python3 life.py seed=1 arm=alone|joined [plan=add:400000,sub:200000] [replay=1] [block=50] [floor=0.01] [cascade=0|1 (casc.py)] [lessons=agree (PLAN_5.md)] [tag=...]
Scores (never-shown lines, whole answer, own bytes, after a read line of the kind named; five draws): additions; subtractions on the clean score
(the never-shown pairs, of the 2,000 never shown, whose line no shown pair writes, as replay6.py)."""
import json, os, random, sys, time
import numpy as np
import both as B
import replay as RP
from both import C, G
from dream import lesson_arrays
from dream2 import mixed_dream

def run(kw):
    seed = int(kw.get("seed", 1)); arm = kw.get("arm", "joined"); block = int(kw.get("block", 50)); floor = float(kw.get("floor", 0.01)); rep = int(kw.get("replay", 1)); t0 = time.process_time()
    plan = [(p.split(":")[0], int(p.split(":")[1])) for p in kw.get("plan", "add:400000,sub:200000").split(",")]; OP = dict(add="+", sub="-")
    rng, PP = B.split(seed); TAB = {k: np.array([C.text(OP[k], a, b) for a, b in PP[OP[k]][2000:]], np.int64) for k in OP}; shown_sub = set(map(tuple, TAB["sub"].tolist()))
    held = dict(add=np.array([C.text("+", a, b) for a, b in PP["+"][:500]]), sub=np.array([C.text("-", a, b) for a, b in PP["-"][:2000] if tuple(C.text("-", a, b)) not in shown_sub]))
    g = np.random.default_rng([seed, 7]); prevs = {k: [TAB[k][g.integers(8000, size=len(held[h]))] for _ in range(5)] for k, h in (("add", "add"), ("sub", "add"))}       # pasts for additions
    prevs_s = {k: [TAB[k][g.integers(8000, size=len(held["sub"]))] for _ in range(5)] for k in ("add", "sub")}                                                          # pasts for subtractions
    if "cascade" in kw: import casc; casc.SysC.CASCADE = int(kw["cascade"]); B.Sys3 = casc.SysC
    total = sum(n for _, n in plan); cs = B.new_counting(seed, total) if arm != "relay" else None; r = B.new_relay(seed) if arm != "alone" else None
    cells = RP.Counting(8); rnd = random.Random(seed * 1000 + 7); srng = np.random.default_rng([seed, 777]); les = kw.get("lessons", "none") == "agree" and arm == "joined"; read = set(); drng = np.random.default_rng([seed, 6262]); lst = dict(composed=0, new=0, new_right=0, new_additions=0, lesson_bytes_taken=0, lesson_bytes_offered=0)
    arms = dict(alone=[("cell1", 1, False), ("cell2", 2, False)], relay=[("relay", "relay", False)],
                joined=[("relay", "relay", False), ("cell1", 1, False), ("cell2", 2, False), ("cell2_relay_silent", 2, True)])[arm]
    def sc(Ls, pv):
        out = {}
        for name, who, mute in arms: a = [B.answer(cs, r, Ls, p, floor, who, mute)[0] for p in pv]; out[name] = float(np.mean(a))
        return out
    def as_replay6(Ls, pool):                                                                          # replay6.py's score_after, for the check of the relay system against that line's files
        tot = 0.0
        for _ in range(5): P = pool[srng.integers(len(pool), size=len(Ls))]; tot += B.answer(None, r, Ls, P, floor, "relay")[0]
        return float(tot / 5)
    tail9 = np.full(9, C.END, np.int64); tail10 = np.full(10, C.END, np.int64); rows = []; out = dict(task="life", args=kw, seed=seed, arm=arm, n_clean=int(len(held["sub"])), rows=rows, ends={}, as_replay6={}); t_rel = 0.0; st = dict(replayed=0, replayed_add=0, wrong=0)
    for pi, (name, n) in enumerate(plan):
        every = 100000 if n > 100000 else n; done = 0
        while done < n:
            k = min(2000 if les else every, n - done); X = TAB[name][rng.integers(8000, size=k)]; done += k
            if r is not None:
                t1 = time.process_time(); flat = np.concatenate([tail10, X.ravel()]); idx = np.arange(k * 10); H = flat[(idx[:, None] + 10) - 1 - np.arange(C.D)[None, :]]; M = np.zeros((k * 10, 14), np.float32); xl = np.full(k * 10, -1, np.int32)
                for b0 in range(0, k, block):
                    b1 = min(k, b0 + block); sl = slice(b0 * 10, b1 * 10); M[sl], xl[sl] = B.message(B.branches(r, H[sl]), floor); lines = []
                    for Lr in X[b0:b1].tolist():
                        lines.append(Lr)
                        if rep:
                            cells.count(Lr); rnd.random(); E = cells.replay(rnd); assert len(E) == 10; lines.append(E); wf, okk, op = RP.kind(E); st["replayed"] += 1; st["replayed_add"] += op == "+"; st["wrong"] += not okk
                    part = np.array(lines, np.int32); r.bx = part.ravel(); r.by = np.concatenate([part[:, 1:], np.full((len(part), 1), -1, np.int32)], 1).ravel(); r.flush()
                xl[::10] = -1; tail10 = flat[-10:]; t_rel += time.process_time() - t1
            if cs is not None:
                x = np.concatenate([tail9, X.ravel()]); cs.give(M, xl) if r is not None else cs.give(None, None); cs.learn(G.ctx_of(x, 9, len(x), 9), x[9:]); cs.give(None, None); tail9 = x[-9:]
            if les:                                                                # PLAN_5.md: lessons by agreement on the new lines the relay system composes (dream2.py, composer=relay sure=1 teacher=agree)
                read.update(map(bytes, X.astype(np.uint8))); dr, L, ended = mixed_dream(cs, r, X, drng, 16, 1); new = ended & ~np.array([bytes(dr[i, :L[i]].astype(np.uint8)) in read for i in range(k)]); lst["composed"] += k; lst["new"] += int(new.sum())
                if new.any():
                    ctx, y = lesson_arrays(X, dr, L, new); seq = np.concatenate([X, dr], 1); i_, tt = np.nonzero(new[:, None] & (np.arange(dr.shape[1])[None, :] < L[:, None])); target = B.branches(r, seq[i_[:, None], (10 + tt)[:, None] - 1 - np.arange(C.D)[None, :]]).argmax(1)
                    cs.give(None, None); cs.cell = 2; take = (tt > 0) & (cs.say(ctx).argmax(1) == target); lst["lesson_bytes_taken"] += int(take.sum()); lst["lesson_bytes_offered"] += int(len(y)); kinds = [RP.kind(dr[q, :L[q]].tolist()) for q in np.flatnonzero(new)]; lst["new_right"] += sum(q[1] for q in kinds); lst["new_additions"] += sum(q[2] == "+" for q in kinds)
                    if take.any(): cs.only_trust = True; cs.give(None, None); cs._run(ctx[take], target[take], True, False); cs.only_trust = False
            if done % every and done < n: continue
            row = dict(phase=pi, name=name, lines=done, add_after_add=sc(held["add"], prevs["add"]))
            if pi > 0: row["add_after_sub"] = sc(held["add"], prevs["sub"]); row["sub_after_sub"] = sc(held["sub"], prevs_s["sub"])
            if cs is not None and r is not None: row["relay_trust_first_cell"] = cs.Tx.tolist(); row["relay_trust_uses"] = cs.Txu.tolist()
            rows.append(row); print(f"seed {seed} {arm} phase {pi} {name} {done}: " + " | ".join(f"{q} " + " ".join(f"{a} {v:.1f}" for a, v in row[q].items()) for q in ("add_after_add", "add_after_sub", "sub_after_sub") if q in row), flush=True)
        out["ends"][f"phase_{pi}"] = dict(add_after_add=sc(held["add"], prevs["add"]), add_after_sub=sc(held["add"], prevs["sub"]), sub_after_sub=sc(held["sub"], prevs_s["sub"]), sub_after_add=sc(held["sub"], prevs_s["add"]))
        if r is not None: out["as_replay6"][f"phase_{pi}"] = dict(first_after_first_task_line=as_replay6(held["add"], TAB["add"]), first_after_second_task_line=as_replay6(held["add"], TAB["sub"]),
                                                                  second_clean_after_second_task_line=as_replay6(held["sub"], TAB["sub"]), second_clean_after_first_task_line=as_replay6(held["sub"], TAB["add"]))
    out.update(lessons=lst, replay=st, cpu=round(time.process_time() - t0, 1), cpu_relay_and_replay=round(t_rel, 1), store_cells=cells.size()[0])
    if cs is not None: out["sizes"] = cs.sizes()
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); res = run(kw); tag = kw.get("tag") or "life_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    os.makedirs(os.path.join(B.HERE, "res"), exist_ok=True); json.dump(res, open(os.path.join(B.HERE, "res", tag + ".json"), "w")); print(tag, "| ends", json.dumps(res["ends"]), "| as replay6", json.dumps(res["as_replay6"]), "| replay", res["replay"], "| cpu", res["cpu"], flush=True)
