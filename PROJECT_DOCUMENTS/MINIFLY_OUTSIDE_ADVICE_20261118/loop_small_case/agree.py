#!/usr/bin/env python3
"""PLAN_10.md, task B: a verb agrees with its subject across words in between. The cells of hold.py (compiled kernel) with a chain of two.
    python3 agree.py <label> text=ag0|ag1 [seed=1 H=256 K=16 eta1=1 cross=0|1 ret=exact|grown chain=2 lines=300000 every=50000 fast=1]
Sentences `the <noun>[s] <words in between> <verb>[s].`: a singular subject takes the verb with s, a plural subject the verb without.
ag0: the words in between are adverbs. ag1: one time in three a word in between is `by the <noun>[s]` with a number of its own.
Scored: the byte after the verb's third letter (s or the full stop), on pairs of noun and verb never shown together."""
import json, os, sys, time
import numpy as np
import hold
HERE = os.path.dirname(os.path.abspath(__file__)); SYM = "abcdefghijklmnopqrstuvwxyz .\n"
NOUNS = "cat dog pig hen cow bat owl ant bee rat".split(); VERBS = "run sit hop nap hum sip tap nod beg rub".split(); ADV = "here still also soon today again twice far".split()

def setup():
    hold.SYMS = [ord(c) for c in SYM]; hold.S = len(SYM); hold.SI = {c: i for i, c in enumerate(hold.SYMS)}; hold.END = hold.SI[10]
def sentence(n, v, plural, fill):
    t = " ".join(["the", NOUNS[n] + ("s" if plural else "")] + fill + [VERBS[v] + ("" if plural else "s")]) + ".\n"
    return [hold.SI[ord(c)] for c in t], len(t) - (3 if plural else 4)                # the line, and where the verb's third letter is
def phrase(rng, plural=None): return "by the " + NOUNS[rng.integers(10)] + ("s" if (rng.integers(2) if plural is None else plural) else "")
def between(rng, tx, k): return [phrase(rng) if tx == "ag1" and rng.integers(3) == 0 else ADV[rng.integers(len(ADV))] for _ in range(k)]

def score(s, items):
    right = 0
    for L, tv in items:
        h = s.h; prev = list(s.prev)
        for t in range(tv + 1):
            h = s.step(h, s.cols(L[t], prev))[3]
            if s.C: prev = [L[t]] + prev[:-1]
        right += int((s.W2 @ h + s.b2).argmax()) == L[tv + 1]
    return 100 * right / len(items)

def run(kw):
    t0 = time.process_time(); setup(); tx = kw.get("text", "ag0"); seed = int(kw.get("seed", 1)); lines = int(kw.get("lines", 300000)); every = int(kw.get("every", 50000))
    rng = np.random.default_rng(seed); P = [(a, b) for a in range(10) for b in range(10)]; rng.shuffle(P); held, shown = P[:20], P[20:]; tr = np.random.default_rng([seed, 7])
    def items(pairs, k, kind=None):                       # fixed test sentences: every pair, both numbers, four fillings
        out = []
        for n, v in pairs:
            for plural in (0, 1):
                for _ in range(4):
                    f = [ADV[tr.integers(len(ADV))] for _ in range(k)]
                    if kind: f = f[:-1] + [phrase(tr, plural if kind == "same" else 1 - plural)]
                    out.append(sentence(n, v, plural, f))
        return out
    tests = {f"never_{k}": items(held, k) for k in (0, 2, 4)}; tests["shown_2"] = items(shown[:20], 2)
    if tx == "ag1": tests["never_2_phrase_same_number"] = items(held, 2, "same"); tests["never_2_phrase_other_number"] = items(held, 2, "other")
    s = hold.Hold(seed, int(kw.get("H", 256)), int(kw.get("K", 16)), float(eval(kw.get("eta1", "1"))), cross=int(kw.get("cross", 0)), ret=kw.get("ret", "exact"), chain=int(kw.get("chain", 2)))
    s.fast = int(kw.get("fast", 1)); out = dict(label=kw["label"], args=kw, rows=[]); n = 0
    for it in range(lines):
        a, b = shown[rng.integers(len(shown))]; L, _ = sentence(a, b, int(rng.integers(2)), between(rng, tx, int(rng.integers(5)))); s.right_acc += s.line(L); n += len(L) - 1
        if (it + 1) % every == 0:
            s.flush(); out["rows"].append(dict(lines=it + 1, bytes_right=100 * s.right_acc / n, **{k: score(s, v) for k, v in tests.items()})); s.right_acc = 0; n = 0
    out["cpu"] = round(time.process_time() - t0, 1); os.makedirs(os.path.join(HERE, "res"), exist_ok=True); json.dump(out, open(os.path.join(HERE, "res", f"agree_{kw['label']}.json"), "w")); return out

if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[2:]); kw["label"] = sys.argv[1]; o = run(kw); r = o["rows"][-1]
    print(o["label"], "| never-shown pairs, words in between 0/2/4: " + " ".join(f"{x['lines'] // 1000}k {x['never_0']:.0f}/{x['never_2']:.0f}/{x['never_4']:.0f}" for x in o["rows"]),
          f"| shown {r['shown_2']:.0f}", (f"| phrase with the same number {r['never_2_phrase_same_number']:.0f}, with the other {r['never_2_phrase_other_number']:.0f}" if "never_2_phrase_same_number" in r else ""), f"| bytes {r['bytes_right']:.0f}", o["cpu"], "s")
