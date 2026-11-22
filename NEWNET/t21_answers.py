"""Whole answers in the structure's own bytes on never-shown combinations (22 November 2026). The learner is net10 unchanged.
Tasks (each its own stream, n_values=128, every line ends in "\n"):
  add2: "a+b=c\n", a, b in 0..99 as two digits, c = a+b as three digits ("07+85=092\n"); 2,000 of the 10,000 pairs never shown;
  copy: "w=w\n", w 3 letters from "abcdefghij" (1,000 strings); 200 never shown;
  rev:  "w=W\n", W = w reversed; the same strings and the same never-shown 200.
The never-shown items are drawn with a fixed seed (2021), the same for every structure and seed.
Life: lines drawn at random from the shown items (seed + 5000), read with learning on (teach: the next byte is given). After every `every`
bytes (at the first line end past it), a test with learning off on up to 500 shown and up to 500 never-shown items (fixed, seed 2022;
copy/rev have only 200 never-shown items, all are used):
  free: feed the line up to and including "=", then the structure answers byte by byte, its own answer fed back as input, until it gives
        "\n" or 6 bytes; the whole answer (all bytes and the newline) is scored; then "\n" is fed if it did not give it itself.
  forced: feed the line up to "=", then at each answer byte the structure's answer is scored and the right byte is fed (teacher-forced).
Places are reported for every answer byte (add2: hundreds, tens, units, newline). The life continues after a test (the test bytes are
read with learning off; nothing is reset).
Usage: python3 t21_answers.py task=add2 struct=K8 seed=1 life=400000 every=100000 out=answers_runs/x.jsonl"""
import sys, json, time, numpy as np
from net10 import Tissue
STRUCT = dict(K8=dict(L=16, cC=64, joint=1, mesh=1, mesh_base=0, mesh_dag=5, c_k_win=4, c_th_on=0.9, c_th_match=0.75),
              N1=dict(L=8, cC=64, M=32, joint=1, mesh=2, mesh_base=0, mesh_dag=2, c_k_win=4))
a = dict(x.split("=", 1) for x in sys.argv[1:])
task, sname, seed = a.get("task", "add2"), a.get("struct", "K8"), int(a.get("seed", 1))
life, every, ntest, out = int(a.get("life", 400000)), int(a.get("every", 100000)), int(a.get("ntest", 500)), a.get("out")
tmax = float(a.get("tmax", 1e9))                                    # stop (and record it) after this many seconds of life

if task == "add2":
    items = [(f"{x:02d}+{y:02d}=", f"{x + y:03d}\n") for x in range(100) for y in range(100)]; nheld = 2000
else:
    al = "abcdefghij"; ws = [p + q + r for p in al for q in al for r in al]
    items = [(w + "=", (w if task == "copy" else w[::-1]) + "\n") for w in ws]; nheld = 200
held = set(np.random.default_rng(2021).choice(len(items), nheld, replace=False).tolist())
shown = [i for i in range(len(items)) if i not in held]; never = sorted(held)
trng = np.random.default_rng(2022)
test_sets = {"shown": trng.choice(shown, min(ntest, len(shown)), replace=False), "never": trng.choice(never, min(ntest, len(never)), replace=False)}
enc = lambda s: [ord(c) for c in s]

net = Tissue(seed=seed, n_values=128, **STRUCT[sname]); rng = np.random.default_rng(seed + 5000)

def test(idx):
    P = len(items[idx[0]][1]); whole = 0; free_place = np.zeros(P); forced_place = np.zeros(P); outs = []
    for i in idx:                                                   # free: its own bytes fed back
        q, ans = items[i]
        for b in enc(q): net.step(b, False)
        o = []
        for _ in range(6):
            g = net.answer(); o.append(g)
            net.step(g, False)
            if g == 10: break
        if o[-1] != 10: net.step(10, False)
        A = enc(ans); whole += o == A
        for k in range(P): free_place[k] += k < len(o) and o[k] == A[k]
        if len(outs) < 8: outs.append(q + "".join(chr(c) if 32 <= c < 127 else "\\x%02x" % c for c in o).replace("\n", "\\n"))
    for i in idx:                                                   # forced: the right bytes fed
        q, ans = items[i]
        for b in enc(q): net.step(b, False)
        for k, b in enumerate(enc(ans)):
            forced_place[k] += net.answer() == b; net.step(b, False)
    n = len(idx)
    return dict(n=n, whole=whole / n, free_place=(free_place / n).round(4).tolist(), forced_place=(forced_place / n).round(4).tolist(),
                forced_bytes=float(forced_place.sum() / (n * P)), examples=outs)

def log(d):
    print(json.dumps(d), flush=True)
    if out: open(out, "a").write(json.dumps(d) + "\n")

log(dict(task=task, struct=sname, seed=seed, options=STRUCT[sname], life=life, every=every, n_items=len(items), n_shown=len(shown),
         n_never=len(never), n_test={k: len(v) for k, v in test_sets.items()}))
nb, nxt, t0, stopped = 0, every, time.time(), None
while nb < life:
    q, ans = items[shown[int(rng.integers(len(shown)))]]
    for b in enc(q + ans): net.step(b, True)
    nb += len(q) + len(ans)
    late = time.time() - t0 > tmax
    if nb >= nxt or nb >= life or late:
        ts = time.time(); r = {k: test(v) for k, v in test_sets.items()}
        log(dict(bytes=nb, life_s=round(ts - t0), test_s=round(time.time() - ts), **r)); nxt += every
        if late: stopped = f"stopped at {nb} bytes: tmax {tmax} s passed"; break
log(dict(done=True, bytes=nb, total_s=round(time.time() - t0), stopped=stopped))
