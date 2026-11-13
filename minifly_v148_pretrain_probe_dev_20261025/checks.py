"""Cycle-1 unit checks for V148. Run: python checks.py [out.json]  (exits 1 on any failure)."""
import json
import math
import pickle
import sys
import time

import numpy as np

import corpus
import ladder
import probe

results = []


def check(name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + ("" if detail == "" else f"  {detail}"), flush=True)


t0 = time.process_time()
# --- corpus
for name in corpus.TEXTS:
    raw = corpus.load(name)
    recs = corpus.records(raw)
    check(f"{name}: SHA-256 pin, {len(recs)} records, max record {max(map(len, recs))} bytes",
          len(recs) > 0 and max(map(len, recs)) <= corpus.MAX_LINE)
    check(f"{name}: records deterministic", recs == corpus.records(corpus.load(name)))
sp = corpus.split(corpus.records(corpus.load("alice")), 40, 5, 10)
o = sp["offsets"]
check("alice split: train < calib < held, disjoint", o["train"][1] == o["calib"][0] and o["calib"][1] <= o["held"][0], o)
pf = corpus.prefixes(sp["train"], [5, 10, 20, 40])
check("nested prefixes increasing and within train", list(pf.values()) == sorted(pf.values()) and
      pf[40] == len(sp["train"]), pf)

# --- ladder: hand-computed toy values and normalisation
toy = [b"ab", b"ac"]
m = ladder.PPMC(2).fit(toy)
EOS = corpus.EOS
check("PPMC2 toy p(b|a) = 1/4", abs(m.prob([97], 98) - 0.25) < 1e-12)
check("PPMC2 toy p(EOS|a) = 1/6 (escape 1/2, order-0 2/6)", abs(m.prob([97], EOS) - 1 / 6) < 1e-12)
check("PPMC2 toy p(unseen|a) = 1/(6*253)", abs(m.prob([97], 122) - 1 / (6 * 253)) < 1e-15)
bg = ladder.Bigram().fit(toy)
check("Bigram toy p(b|a) = 0.5*0.5 + 0.5*2/263", abs(bg.prob([97], 98) - (0.25 + 0.5 * 2 / 263)) < 1e-12)
dev = corpus.records(corpus.load("buster"))[:200]
rng = np.random.default_rng(0)
for mod in ladder.ladder():
    mod.fit(dev)
    worst = 0.0
    for _ in range(20):
        r = dev[rng.integers(len(dev))]
        cut = int(rng.integers(0, len(r) + 1))
        ctx = list(r[:cut])
        worst = max(worst, abs(sum(mod.prob(ctx, x) for x in range(257)) - 1.0))
    check(f"{mod.name}: probabilities sum to 1 over 257 symbols (max error {worst:.2e})", worst < 1e-9)

# --- MiniFly probe
mem = probe.new_memory()
recs = corpus.records(corpus.load("buster"))
for r in recs[:5]:
    mem.train(b"", r)
h0 = probe.state_hash(mem)
L1 = probe.line_logits(mem, recs[6])
check("line_logits shape = len(line)+1 rows x 257", L1.shape == (len(recs[6]) + 1, 257), L1.shape)
check("probe leaves memory state unchanged", probe.state_hash(mem) == h0)
L2 = probe.line_logits(mem, recs[6])
check("probe deterministic (two reads identical)", np.array_equal(L1, L2))
g, why = probe.generate(mem, recs[6][:10], cap=8)
check("generation leaves memory state unchanged", probe.state_hash(mem) == h0, {"generated_hex": g.hex(), "stop": why})
Lc, Yc = probe.collect(mem, recs[6:8])
T, bpb = probe.calibrate(Lc, Yc)
check("temperature calibration returns finite bits", math.isfinite(bpb) and T > 0, {"T": T, "bpb": round(bpb, 4)})
blob = pickle.dumps(mem)
mem2 = pickle.loads(blob)
check("pickle round trip: same state hash and same logits",
      probe.state_hash(mem2) == h0 and np.array_equal(probe.line_logits(mem2, recs[6]), L1))

# --- cost on ~1 KB of dev text
tt = time.process_time()
n_ev = 0
for r in recs[5:25]:
    mem.train(b"", r)
    n_ev += len(r) + 1
train_s = (time.process_time() - tt) / n_ev
tp = time.process_time()
Lc, Yc = probe.collect(mem, recs[25:35])
probe_s = (time.process_time() - tp) / len(Yc)
check("cost measured", True, {"train_cpu_s_per_event": round(train_s, 4), "probe_cpu_s_per_symbol": round(probe_s, 4),
                              "train_events": n_ev, "probe_symbols": int(len(Yc))})

ok = all(r["pass"] for r in results)
json.dump({"pass": ok, "cpu_s": round(time.process_time() - t0, 1), "checks": results},
          open(sys.argv[1] if len(sys.argv) > 1 else "cycles/CYCLE1_CHECKS.json", "w"), indent=1)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
