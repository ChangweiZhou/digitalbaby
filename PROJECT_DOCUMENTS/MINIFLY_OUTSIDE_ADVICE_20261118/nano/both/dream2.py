#!/usr/bin/env python3
"""nano/both, PLAN_2.md addition 2: lessons on new lines, composed by the counting system and answered by the relay system.
The joined system of both.py reads a chunk of real lines. Then the counting system composes one line after each of them: each byte drawn from its
second answering cell's shares given what stands before (the relay system silent), until a line end or sixteen bytes. The composed lines that the
system has not read are lessons. At each byte of a lesson the relay system gives its answer (its largest branch) for what stands before; the two
answering cells change their trusts and parts towards that answer and nothing else (no cell counts; the relay system is silent as a sender).
A lesson byte is taken with a chance equal to the teacher's edge at its level of sureness: twice the share of stream bytes at that level on which the
relay system's answer was the byte that came (its own record, kept from birth, each answer formed before the byte was learned), less one; nothing if below.
    python3 dream2.py seed=1 teacher=relay|self|none [edge=1] [lines=400000] [chunk=2000] [tag=...]
teacher=self (the control): the target is the counting system's own answer (second cell's largest share) with the same chances.  edge=0: every lesson byte is taken."""
import json, os, sys, time
import numpy as np
import both as B
import casc
import replay as RP
from both import C, G
from dream import lesson_arrays

def counting_dream(cs, pasts, rng, maxlen=16):
    n = len(pasts); seq = np.concatenate([np.asarray(pasts, np.int64), np.full((n, maxlen), C.END, np.int64)], 1); done = np.zeros(n, bool); L = np.zeros(n, np.int64); cs.give(None, None); cs.cell = 2
    for t in range(maxlen):
        p = cs.say(seq[:, 10 + t - 1 - np.arange(cs.D)]); v = np.minimum((np.cumsum(p, 1) < rng.random(n)[:, None]).sum(1), p.shape[1] - 1); seq[:, 10 + t] = np.where(done, C.END, v); new = (~done) & (v == C.END); L[new] = t + 1; done |= new
        if done.all(): break
    return seq[:, 10:], L, done

def run(kw):
    seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 400000)); teacher = kw.get("teacher", "relay"); chunk = int(kw.get("chunk", 2000)); use_edge = int(kw.get("edge", 1)); block = 50; floor = 0.01; t0 = time.process_time()
    casc.install(0); rng, PP = B.split(seed); held = PP["+"][:500]; shown = PP["+"][2000:]; TAB = np.array([C.text("+", a, b) for a, b in shown], np.int64); Ls = np.array([C.text("+", a, b) for a, b in held])
    g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]; drng = np.random.default_rng([seed, 6262])
    marks = sorted(set([m for m in (5000, 10000, 20000, 40000) if m <= lines] + list(range(50000, lines + 1, 50000)) + [lines]))
    cs = B.new_counting(seed, lines); r = B.new_relay(seed); read = set(); hits = np.zeros(10); uses = np.zeros(10)
    arms = [("relay", "relay", False), ("cell1", 1, False), ("cell2", 2, False), ("cell2_relay_silent", 2, True)]
    tail9 = np.full(9, C.END, np.int64); tail10 = np.full(10, C.END, np.int64); done = 0; rows = []; keys = ("composed", "with_line_end", "new", "new_well_formed", "new_right", "lesson_bytes_offered", "lesson_bytes_taken", "teacher_right_on_answer_bytes", "answer_bytes_taken"); st = {q: 0 for q in keys}; seg = dict(st)
    while done < lines:
        k = min(chunk, lines - done, next(m for m in marks if m > done) - done); X = TAB[rng.integers(8000, size=k)]; done += k
        M, xl, tail10 = B.relay_read(r, X, tail10, block, floor); x = np.concatenate([tail9, X.ravel()]); cs.give(M, xl); cs.learn(G.ctx_of(x, 9, len(x), 9), x[9:]); cs.give(None, None); tail9 = x[-9:]
        sp = xl >= 0; np.add.at(uses, xl[sp], 1); np.add.at(hits, xl[sp], (M[sp].argmax(1) == X.ravel()[sp]).astype(float)); read.update(map(bytes, X.astype(np.uint8)))
        if teacher != "none":
            dr, L, ended = counting_dream(cs, X, drng); new = ended & ~np.array([bytes(dr[i, :L[i]].astype(np.uint8)) in read for i in range(k)]); kinds = [RP.kind(dr[i, :L[i]].tolist()) for i in np.flatnonzero(new)]
            upd = dict(composed=k, with_line_end=int(ended.sum()), new=int(new.sum()), new_well_formed=sum(q[0] for q in kinds), new_right=sum(q[1] for q in kinds))
            if new.any():
                ctx, y = lesson_arrays(X, dr, L, new); seq = np.concatenate([X, dr], 1); i, tt = np.nonzero(new[:, None] & (np.arange(dr.shape[1])[None, :] < L[:, None])); H = seq[i[:, None], (10 + tt)[:, None] - 1 - np.arange(C.D)[None, :]]
                a = B.branches(r, H); _, lev = B.message(a, floor); edge = np.maximum(0.0, 2 * hits / np.maximum(uses, 1) - 1) if use_edge else np.ones(10); take = (tt > 0) & (drng.random(len(y)) < edge[lev])
                if teacher == "relay": target = a.argmax(1)
                else: cs.give(None, None); cs.cell = 2; target = cs.say(ctx).argmax(1)
                true_line = np.array([q[1] for q in kinds])[np.cumsum(new)[i] - 1]; ansb = take & (tt >= 6) & (tt <= 8); upd.update(lesson_bytes_offered=int(len(y)), lesson_bytes_taken=int(take.sum()), answer_bytes_taken=int(ansb.sum()), teacher_right_on_answer_bytes=int((ansb & true_line & (target == y)).sum()))
                if take.any(): cs.only_trust = True; cs.give(None, None); cs._run(ctx[take], target[take], True, False); cs.only_trust = False
            for d in (st, seg):
                for q, v in upd.items(): d[q] += v
        if done in marks:
            row = dict(lines=done); row.update(B.reading(cs, r, Ls, prevs, floor, arms)); row["relay_trust_first_cell"] = cs.Tx.tolist(); row["teacher_edge_by_level"] = np.maximum(0.0, 2 * hits / np.maximum(uses, 1) - 1).tolist(); row["since_last_reading"] = seg; seg = {q: 0 for q in keys}; rows.append(row); s_ = row["since_last_reading"]
            print(f"seed {seed} teacher {teacher} {done}: " + " ".join(f"{q} {v['after']:.1f}" for q, v in row.items() if isinstance(v, dict) and "after" in v) + " | only counts from 8/64/512 " + " ".join(f"{row[f'look_cell2_only_counts_from_{th}']:.1f}" for th in (8, 64, 512))
                  + f" | class kinds alone {row['look_cell2_class_kinds_alone']:.1f} | composed {s_['composed']} new {s_['new']} well formed {s_['new_well_formed']} right {s_['new_right']} | lesson bytes taken {s_['lesson_bytes_taken']} of {s_['lesson_bytes_offered']} | edge " + " ".join(f"{e:.2f}" for e in row["teacher_edge_by_level"]), flush=True)
    return dict(task="dream2", args=kw, seed=seed, teacher=teacher, rows=rows, totals=st, sizes=cs.sizes(), cpu=round(time.process_time() - t0, 1))

def check():
    """The counting system's composed bytes are drawn from its second cell's shares (never a byte of no share), and the lesson contexts are those read byte by byte."""
    rng = np.random.default_rng(3); lines = np.array([C.text("+", int(a), int(b)) for a, b in rng.integers(0, 100, (600, 2))]); cs = B.new_counting(3, 600); x = np.concatenate([np.full(9, C.END), lines.ravel()]); cs.learn(G.ctx_of(x, 9, len(x), 9), x[9:])
    dr, L, ended = counting_dream(cs, lines[:100], np.random.default_rng(1)); ctx, y = lesson_arrays(lines[:100], dr, L, ended); cs.cell = 2; P = cs.say(ctx); low = float(P[np.arange(len(y)), y].min()); pos = 0; flat_ok = True
    for i in np.flatnonzero(ended):
        seq = lines[i].tolist() + dr[i].tolist()
        for t in range(int(L[i])): flat_ok &= (ctx[pos].tolist() == seq[10 + t - 9:10 + t][::-1]) and y[pos] == seq[10 + t]; pos += 1
    wf = sum(RP.kind(dr[i, :L[i]].tolist())[0] for i in np.flatnonzero(ended)); good = low > 0 and flat_ok and ended.mean() > 0.5
    print(f"the counting system's composed lines: {int(ended.sum())} of 100 reach a line end, {wf} well formed; smallest share of a composed byte under the second cell {low:.4f}; contexts equal those read byte by byte: {flat_ok}", "PASSED" if good else "FAILED"); return good

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[1:]); res = run(kw); tag = kw.get("tag") or "dream2_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    json.dump(res, open(os.path.join(B.HERE, "res", tag + ".json"), "w")); print(tag, "| totals", res["totals"], "| cpu", res["cpu"], flush=True)
