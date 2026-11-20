#!/usr/bin/env python3
"""nano/both, PLAN_4.md: the reverse help. The counting system (nano/merge's cells with classes and the two answering cells, alone: the relay system is
not a sender here) reads the stream; the relay system reads the same stream and, after each real line, learns one more line:
    mode=composed: a line the counting system composes: two numbers drawn by chance among the pairs not met so far (the program's own record of the
                   pairs it has shown), with the counting system's own answer (its second cell, own bytes, read after the real line just read);
    mode=shown:    the control: a line drawn by chance from the lines the program shows;
    mode=none:     nothing more (the relay system alone, as both.py).
    python3 teach.py seed=1 mode=composed|shown|none [lines=40000] [excl=1] [chunk=500] [tag=...]
excl=1: the 500 pairs that are scored are never composed (so that the score stays a score of lines the relay system has not met in any form);
excl=0: any pair not met may be composed, the scored ones among them (then the relay system has met scored pairs with the counting system's answers).
The counting system reads a chunk of `chunk` real lines, then composes one line for each of them (so its answers are at most a chunk newer than the
real line they follow); the relay system then learns the chunk, real and added lines in turn."""
import json, os, sys, time
import numpy as np
import both as B
from both import C, G

def own_answer(cs, Ls, prev):
    """The counting system's second cell writes the three answer bytes of each line of Ls in its own bytes, after the line prev."""
    own = Ls.copy(); cs.give(None, None); cs.cell = 2
    for t in (6, 7, 8): own[:, t] = cs.say(B.past_of(own, prev, t, cs.D)).argmax(1)
    return own

def run(kw):
    seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 40000)); mode = kw.get("mode", "composed"); excl = int(kw.get("excl", 1)); chunk = int(kw.get("chunk", 500)); t0 = time.process_time()
    rng, PP = B.split(seed); held = PP["+"][:500]; shown = PP["+"][2000:]; TAB = np.array([C.text("+", a, b) for a, b in shown], np.int64); Ls = np.array([C.text("+", a, b) for a, b in held])
    g = np.random.default_rng([seed, 7]); prevs = [TAB[g.integers(8000, size=500)] for _ in range(5)]; trng = np.random.default_rng([seed, 4242])
    cs = B.new_counting(seed, lines); r = B.new_relay(seed); met = np.zeros(10000, bool); scored = np.zeros(10000, bool); scored[[a * 100 + b for a, b in held]] = True
    marks = [m for m in (5000, 10000, 20000, 40000) if m < lines] + [lines]; tail9 = np.full(9, C.END, np.int64); done = 0; rows = []; st = dict(added=0, wrong=0, added_scored_pairs=0); wrong_lines = {}; seg = dict(added=0, wrong=0)
    def score():
        rel = [B.answer(None, r, Ls, pv, 0.01, "relay") for pv in prevs]; cnt = [B.answer(cs, None, Ls, pv, 0.01, 2) for pv in prevs]
        return dict(relay=float(np.mean([q[0] for q in rel])), relay_place=np.mean([q[1] for q in rel], 0).tolist(), counting_cell2=float(np.mean([q[0] for q in cnt])))
    while done < lines:
        k = min(chunk, lines - done, next(m for m in marks if m > done) - done); X = TAB[rng.integers(8000, size=k)]; done += k
        x = np.concatenate([tail9, X.ravel()]); cs.give(None, None); cs.learn(G.ctx_of(x, 9, len(x), 9), x[9:]); tail9 = x[-9:]; met[X[:, 0] * 1000 + X[:, 1] * 100 + X[:, 3] * 10 + X[:, 4]] = True
        if mode == "composed":
            cand = np.flatnonzero(~met & ~scored) if excl else np.flatnonzero(~met); q = cand[trng.integers(len(cand), size=k)]; true = np.array([C.text("+", int(v) // 100, int(v) % 100) for v in q]); E = own_answer(cs, true, X)
            bad = (E != true).any(1); st["wrong"] += int(bad.sum()); seg["wrong"] += int(bad.sum()); st["added_scored_pairs"] += int(scored[q].sum())
            for v, e in zip(q[bad].tolist(), E[bad].tolist()): wrong_lines[v] = e
        elif mode == "shown": E = TAB[trng.integers(8000, size=k)]
        else: E = None
        part = (np.stack([X, E], 1).reshape(-1, 10) if E is not None else X).astype(np.int32); st["added"] += 0 if E is None else k; seg["added"] += 0 if E is None else k
        r.bx = part.ravel(); r.by = np.concatenate([part[:, 1:], np.full((len(part), 1), -1, np.int32)], 1).ravel(); r.flush()
        if done in marks:
            row = dict(lines=done, **score(), added_since_last=seg["added"], wrong_since_last=seg["wrong"], pairs_not_met=int((~met).sum())); seg = dict(added=0, wrong=0); rows.append(row)
            print(f"seed {seed} {mode} excl {excl} {done}: relay {row['relay']:.1f} counting {row['counting_cell2']:.1f} | added {row['added_since_last']} wrong {row['wrong_since_last']}", flush=True)
    out = dict(task="teach", args=kw, seed=seed, mode=mode, rows=rows, totals=st, cpu=round(time.process_time() - t0, 1))
    if wrong_lines:                                                                    # what the relay system does with the pairs whose composed line was wrong (the last wrong line of each): read after a shown line
        q = np.array(sorted(wrong_lines)); true = np.array([C.text("+", int(v) // 100, int(v) % 100) for v in q]); comp = np.array([wrong_lines[int(v)] for v in q]); own = true.copy(); pv = TAB[trng.integers(8000, size=len(q))]
        for t in (6, 7, 8): own[:, t] = B.branches(r, B.past_of(own, pv, t, C.D)).argmax(1)
        out["pairs_composed_wrongly"] = dict(pairs=int(len(q)), relay_gives_the_true_answer=float(100 * (own == true).all(1).mean()), relay_gives_the_composed_wrong_answer=float(100 * (own == comp).all(1).mean()))
        cn = own_answer(cs, true, pv); out["pairs_composed_wrongly"]["counting_gives_the_true_answer_at_the_end"] = float(100 * (cn == true).all(1).mean())
    return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:]); res = run(kw); tag = kw.get("tag") or "teach_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items()))
    json.dump(res, open(os.path.join(B.HERE, "res", tag + ".json"), "w")); print(tag, "| totals", res["totals"], "|", res.get("pairs_composed_wrongly"), "| cpu", res["cpu"], flush=True)
