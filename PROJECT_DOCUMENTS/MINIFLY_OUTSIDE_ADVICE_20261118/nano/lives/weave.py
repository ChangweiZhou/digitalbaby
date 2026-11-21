#!/usr/bin/env python3
"""nano/lives, PLAN_10.md: a woven life. The six bodies are cut into blocks of 65,536 bytes and read in rounds, one block of each stream in an order drawn once.
    python3 weave.py life sys=ZA|B|MC|MCB|BA|MCBA order=dev|seed:73129 tag=..      (B: remembered trusts; A: with the waking of PLAN_8.md)
    python3 weave.py single sys=ZA|B|MC|MCB|BA|MCBA stream=py                        the same system reading one stream alone, scored over the same bytes
    python3 weave.py report
Every system runs on awake.py's kernel, with the parts switched off that it does not use."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lives as LV, awake as AW
BLOCK = 65536; m = LV.m; NAMES = tuple(LV.NAMES)
SYS = dict(ZA=("ZA", 0, 0), B=("ZA", 65536, 0), MC=("MC", 0, 0), MCB=("MC", 65536, 0), BA=("ZA", 65536, 1), MCBA=("MC", 65536, 1))

def weave(order):
    """The woven array: rounds of one block of each stream; then, for each stream, the last LEAD bytes of its body followed by its test. Returns the array, the list of (stream, place in the array, place in its body, length), and the tests' places."""
    raws = {n: LV.raw(n) for n in order}; off = {n: 0 for n in order}; parts = []; chunks = []; pos = 0
    while any(off[n] < raws[n][1] for n in order):
        for n in order:
            x, cut = raws[n]
            if off[n] >= cut: continue
            k = min(BLOCK, cut - off[n]); parts.append(x[off[n]:off[n] + k]); chunks.append((n, pos, off[n], k)); pos += k; off[n] += k
    test = {}
    for n in NAMES: x, cut = LV.raw(n); parts += [x[cut - LV.LEAD:cut], x[cut:]]; test[n] = (pos + LV.LEAD, pos + LV.LEAD + len(x) - cut); pos += LV.LEAD + len(x) - cut
    return np.concatenate(parts), chunks, test

def score(s, lo, hi, learn):
    n1, n2, mx, top = s.run(lo, hi, learn); return dict(cell2=m(n2), mix=m(mx), share_on_living=float((top == 0).mean()))

def run_life(kw):
    name = kw["sys"]; base, N, awake = SYS[name]; tag = kw["tag"]; order = list(NAMES) if kw["order"] == "dev" else [NAMES[i] for i in np.random.default_rng(int(kw["order"][5:])).permutation(len(NAMES))]
    x, chunks, test = weave(order); s = AW.make(base, x, N, awake); tmp = os.path.join(HERE, "tmp", tag); os.makedirs(tmp, exist_ok=True); t0 = time.time(); end = chunks[-1][1] + chunks[-1][3]; n1, n2, mx, top = s.run(6, end, True)
    read = {}
    for n in order:
        idx = np.concatenate([np.arange(p, p + k) for nn, p, o, k in chunks if nn == n]) - 6; idx = idx[idx >= 0]; h = len(idx) // 2
        read[n] = dict(cell2=float(n2[idx].mean()), mix=float(mx[idx].mean()), second_half=dict(cell2=float(n2[idx[h:]].mean()), mix=float(mx[idx[h:]].mean())), share_on_living=float((top[idx] == 0).mean()), bytes=int(len(idx)))
    tests = {}
    for n in NAMES:
        r = {}
        for learn in (0, 1): p = os.path.join(tmp, f"{n}_{learn}.json"); LV.wait(LV.forked(lambda: score(s, test[n][0], test[n][1], bool(learn)), p)); r["going_on" if learn else "nothing_changing"] = json.load(open(p))
        tests[n] = r
    out = dict(task="weave", tag=tag, sys=name, order=order, blocks=len(chunks), reading=read, tests=tests, sizes=s.sizes(), wall=round(time.time() - t0, 1)); json.dump(out, open(os.path.join(HERE, "res", f"weave__{tag}.json"), "w")); return out

def run_single(kw):
    """The same system reading one stream alone: its body in one piece; scored over the whole body, over the same second half of the body's bytes as a woven life uses, and on its test."""
    name = kw["sys"]; base, N, awake = SYS[name]; n = kw["stream"]; x, body, test = LV.layout([n], [n]); s = AW.make(base, x, N, awake); tmp = os.path.join(HERE, "tmp", f"weave_single_{name}_{n}"); os.makedirs(tmp, exist_ok=True); n1, n2, mx, top = s.run(6, body[n][1], True); h = (len(n2)) // 2; tests = {}
    for learn in (0, 1): p = os.path.join(tmp, f"{learn}.json"); LV.wait(LV.forked(lambda: score(s, test[n][0], test[n][1], bool(learn)), p)); tests["going_on" if learn else "nothing_changing"] = json.load(open(p))
    out = dict(task="weave_single", sys=name, stream=n, reading=dict(cell2=m(n2), mix=m(mx), second_half=dict(cell2=m(n2[h:]), mix=m(mx[h:]))), tests=tests, sizes=s.sizes()); json.dump(out, open(os.path.join(HERE, "res", f"weave_single__{name}_{n}.json"), "w")); return out

def report():
    L = lambda f: json.load(open(os.path.join(HERE, "res", f + ".json"))); have = lambda f: os.path.exists(os.path.join(HERE, "res", f + ".json")); orders = ("dev", "seed73129", "seed93683", "seed116089")
    S = {s: {n: L(f"weave_single__{s}_{n}") for n in NAMES} for s in SYS if all(have(f"weave_single__{s}_{n}") for n in NAMES)}
    for s, d in S.items(): print(f"singles, {s}: reading, second half of the body: " + " ".join(f"{d[n]['reading']['second_half']['mix']:.4f}" for n in NAMES) + " | test going on: " + " ".join(f"{d[n]['tests']['going_on']['mix']:.4f}" for n in NAMES))
    for o in orders:
        rows = [(s, L(f"weave__{s}_{o}")) for s in SYS if have(f"weave__{s}_{o}") and s in S]
        if not rows: continue
        print(f"\n**{o}: {', '.join(rows[0][1]['order'])}** ({rows[0][1]['blocks']} blocks)\n\n| system | reading, second half of each stream's blocks: " + " | ".join(NAMES) + " | mean | less its singles | test at the end, going on: mean | less its singles |\n|---|" + "---|" * 10)
        for s, d in rows:
            for key, label in (("cell2", "the living set"), ("mix", "the mixture")):
                if key == "mix" and SYS[s][1] == 0: continue
                r = [d["reading"][n]["second_half"][key] for n in NAMES]; r0 = [S[s][n]["reading"]["second_half"][key] for n in NAMES]; t = [d["tests"][n]["going_on"][key] for n in NAMES]; t0 = [S[s][n]["tests"]["going_on"][key] for n in NAMES]
                print(f"| {s}, {label} | " + " | ".join(f"{v:.4f}" for v in r) + f" | {np.mean(r):.4f} | {np.mean(r) - np.mean(r0):+.4f} | {np.mean(t):.4f} | {np.mean(t) - np.mean(t0):+.4f} |")
        for s, d in rows:
            if SYS[s][1]: g = [d["reading"][n]["second_half"]["cell2"] - d["reading"][n]["second_half"]["mix"] for n in NAMES]; print(f"* {s}: the mixture below the living set while reading (second half) by " + " ".join(f"{v:+.4f}" for v in g) + f"; by 0.005 or more on {sum(v >= 0.005 for v in g)} of 6; share of the bytes at which the living set held the largest share: " + " ".join(f"{d['reading'][n]['share_on_living']:.2f}" for n in NAMES))

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True); cmd = sys.argv[1]; kw = dict(a.split("=", 1) for a in sys.argv[2:])
    if cmd == "report": report()
    elif cmd == "single":
        if not os.path.exists(os.path.join(HERE, "res", f"weave_single__{kw['sys']}_{kw['stream']}.json")): out = run_single(kw); print("single", kw["sys"], kw["stream"], "| reading", round(out["reading"]["mix"], 4), "second half", round(out["reading"]["second_half"]["mix"], 4), "| test going on", round(out["tests"]["going_on"]["mix"], 4), flush=True)
    elif cmd == "life":
        if not os.path.exists(os.path.join(HERE, "res", f"weave__{kw['tag']}.json")): out = run_life(kw); print(kw["tag"], "order", ",".join(out["order"]), "| reading, second half, the living set:", {n[:2]: round(v["second_half"]["cell2"], 4) for n, v in out["reading"].items()}, "| the mixture:", {n[:2]: round(v["second_half"]["mix"], 4) for n, v in out["reading"].items()}, "| wall", out["wall"], flush=True)
