#!/usr/bin/env python3
"""nano/both, PLAN_2.md addition 1: lessons on new lines. The joined system of both.py reads the stream; after each real line the relay system composes
one line (each byte drawn in proportion to its branches cut off below at nothing, until a line end or sixteen bytes); the composed lines that the
system has not read (lessons=new) are read by the two answering cells, which change their trusts and parts on them and nothing else: no cell counts
them, and the relay system is silent as a sender on them.  lessons=read (the control): the composed lines that the system has read.  lessons=none.
    python3 dream.py seed=1 lessons=new|read|none [arm=joined|alone] [lines=400000] [chunk=500] [tag=...]
arm=alone: the relay system composes and is never a sender (it reads the stream only to be able to compose).
The system reads a chunk of `chunk` real lines, then the relay system composes one line after each of them (its strengths are those at the end of the
chunk), then the lessons are read."""
import json, os, sys, time
import numpy as np
import both as B
import casc
import replay as RP
from both import C, G

def relay_dream(r, pasts, rng, maxlen=16):
    """One composed line after each row of pasts (the ten bytes before it, oldest first). Returns the bytes (n by maxlen), the length to the line end, and whether a line end came."""
    n = len(pasts); seq = np.concatenate([np.asarray(pasts, np.int64), np.full((n, maxlen), C.END, np.int64)], 1); done = np.zeros(n, bool); L = np.zeros(n, np.int64)
    for t in range(maxlen):
        p = np.maximum(B.branches(r, seq[:, 10 + t - 1 - np.arange(C.D)]), 0.0); z = p.sum(1); p = np.where(z[:, None] > 0, p / np.maximum(z, 1e-300)[:, None], 1.0 / p.shape[1])
        v = np.minimum((np.cumsum(p, 1) < rng.random(n)[:, None]).sum(1), p.shape[1] - 1); seq[:, 10 + t] = np.where(done, C.END, v); new = (~done) & (v == C.END); L[new] = t + 1; done |= new
    return seq[:, 10:], L, done

def lesson_arrays(pasts, dreams, L, keep, D=9):
    """Contexts (D delays, newest first) and bytes of the kept composed lines, each read after its own past."""
    seq = np.concatenate([np.asarray(pasts, np.int64), dreams], 1); n, m = dreams.shape; t = np.arange(m)[None, :]; ok = keep[:, None] & (t < L[:, None]); i, tt = np.nonzero(ok)
    return seq[i[:, None], (10 + tt)[:, None] - 1 - np.arange(D)[None, :]], seq[i, 10 + tt]

def run(kw):
    seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 400000)); arm = kw.get("arm", "joined"); les = kw.get("lessons", "new"); chunk = int(kw.get("chunk", 500)); block = 50; floor = 0.01; t0 = time.process_time()
    casc.install(0); rng, PP = B.split(seed); held = PP["+"][:500]; shown = PP["+"][2000:]; TAB = np.array([C.text("+", a, b) for a, b in shown], np.int64); Ls = np.array([C.text("+", a, b) for a, b in held])
    g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]; drng = np.random.default_rng([seed, 5151])
    marks = sorted(set([m for m in (5000, 10000, 20000, 40000) if m <= lines] + list(range(50000, lines + 1, 50000)) + [lines]))
    cs = B.new_counting(seed, lines); r = B.new_relay(seed); sender = arm == "joined"; read = set()
    arms = [("relay", "relay", False), ("cell1", 1, not sender), ("cell2", 2, not sender)] + ([("cell2_relay_silent", 2, True)] if sender else [])
    tail9 = np.full(9, C.END, np.int64); tail10 = np.full(10, C.END, np.int64); done = 0; rows = []; st = dict(composed=0, with_line_end=0, new=0, lessons=0, lesson_bytes=0, new_well_formed=0, new_right=0, read_again=0); seg = dict(st)
    while done < lines:
        k = min(chunk, lines - done, next(m for m in marks if m > done) - done); X = TAB[rng.integers(8000, size=k)]; done += k
        M, xl, tail10 = B.relay_read(r, X, tail10, block, floor); x = np.concatenate([tail9, X.ravel()]); cs.give(M, xl) if sender else cs.give(None, None); cs.learn(G.ctx_of(x, 9, len(x), 9), x[9:]); cs.give(None, None); tail9 = x[-9:]
        read.update(map(bytes, X.astype(np.uint8)))
        if les != "none":
            dr, L, ended = relay_dream(r, X, drng); isread = np.array([bytes(dr[i, :L[i]].astype(np.uint8)) in read for i in range(k)]); new = ended & ~isread; keep = new if les == "new" else (ended & isread)
            for d in (st, seg):
                d["composed"] += k; d["with_line_end"] += int(ended.sum()); d["new"] += int(new.sum()); d["read_again"] += int((ended & isread).sum()); d["lessons"] += int(keep.sum()); d["lesson_bytes"] += int(L[keep].sum())
                kinds = [RP.kind(dr[i, :L[i]].tolist()) for i in np.flatnonzero(new)]; d["new_well_formed"] += sum(q[0] for q in kinds); d["new_right"] += sum(q[1] for q in kinds)
            if keep.any(): ctx, y = lesson_arrays(X, dr, L, keep); cs.only_trust = True; cs.give(None, None); cs._run(ctx, y, True, False); cs.only_trust = False
        if done in marks:
            row = dict(lines=done); row.update(B.reading(cs, r, Ls, prevs, floor, arms)); row["relay_trust_first_cell"] = cs.Tx.tolist(); row["since_last_reading"] = seg; seg = {q: 0 for q in st}; rows.append(row)
            print(f"seed {seed} {arm} lessons {les} {done}: " + " ".join(f"{q} {v['after']:.1f}" for q, v in row.items() if isinstance(v, dict) and "after" in v) + " | only counts from 8/64/512 " + " ".join(f"{row[f'look_cell2_only_counts_from_{th}']:.1f}" for th in (8, 64, 512))
                  + f" | class kinds alone {row['look_cell2_class_kinds_alone']:.1f} | composed {row['since_last_reading']['composed']} new {row['since_last_reading']['new']} of them well formed {row['since_last_reading']['new_well_formed']} right {row['since_last_reading']['new_right']}", flush=True)
    return dict(task="dream", args=kw, seed=seed, arm=arm, lessons=les, rows=rows, totals=st, sizes=cs.sizes(), cpu=round(time.process_time() - t0, 1))

def check():
    """The composed lines are drawn from the stated shares; the lesson arrays are the stream's contexts; a lesson changes trusts and parts and no count."""
    rng = np.random.default_rng(3); r = B.new_relay(3); lines = np.array([C.text("+", int(a), int(b)) for a, b in rng.integers(0, 100, (400, 2))]); B.relay_read(r, lines, np.full(10, C.END), 50, 0.01)
    dr, L, ended = relay_dream(r, lines[:200], np.random.default_rng(1)); ok = True
    # each byte of a composed line has a share above nothing under the relay system's branches for the bytes before it
    bad = 0
    for i in range(200):
        seq = lines[i].tolist() + dr[i].tolist()
        for t in range(int(L[i]) if ended[i] else 16):
            a = r.fwd(seq[10 + t - 10:10 + t][::-1])[3]; bad += int(a[seq[10 + t]] <= 0 and (np.maximum(a, 0).sum() > 0))
    ctx, y = lesson_arrays(lines[:200], dr, L, ended); flat_ok = True; pos = 0
    for i in np.flatnonzero(ended):
        seq = lines[i].tolist() + dr[i].tolist()
        for t in range(int(L[i])): flat_ok &= (ctx[pos].tolist() == seq[10 + t - 9:10 + t][::-1]) and y[pos] == seq[10 + t]; pos += 1
    cs = B.new_counting(3, 400); x = np.concatenate([np.full(9, C.END), lines.ravel()]); cs.learn(G.ctx_of(x, 9, len(x), 9), x[9:]); C0, N0, T0, G0, top0 = cs.C.copy(), cs.N.copy(), cs.T.copy(), cs.Gt.copy(), cs.top
    cs.only_trust = True; cs._run(ctx, y, True, False); cs.only_trust = False; same = bool((cs.C == C0).all() and (cs.N == N0).all() and cs.top == top0); moved = float(np.abs(cs.T - T0).max()), float(np.abs(cs.Gt - G0).max())
    good = bad == 0 and flat_ok and pos == len(y) and same and moved[0] > 0 and moved[1] > 0 and ended.mean() > 0.5
    print(f"composed lines: {int(ended.sum())} of 200 reach a line end; bytes drawn with no share {bad}; lesson arrays equal the contexts read byte by byte: {flat_ok} ({pos} bytes); after the lessons the counts and cells are unchanged: {same}; largest change of a trust {moved[0]:.3f}, of a part {moved[1]:.3f}", "PASSED" if good else "FAILED"); return good

if __name__ == "__main__":
    if sys.argv[1] == "check": sys.exit(0 if check() else 1)
    kw = dict(a.split("=") for a in sys.argv[1:]); res = run(kw); tag = kw.get("tag") or "dream_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    json.dump(res, open(os.path.join(B.HERE, "res", tag + ".json"), "w")); print(tag, "| totals", res["totals"], "| cpu", res["cpu"], flush=True)
