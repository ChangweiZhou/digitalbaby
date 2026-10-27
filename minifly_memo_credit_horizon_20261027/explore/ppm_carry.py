# Corrected 27 Oct 2026 (see MEMO erratum): the first version summed (bits, symbols) before dividing, adding 1 bit.
# Exploratory (S0, evaluator-only, no learning records): counting baselines for the carry-over readings of V163-V167.
# PPM-C (V148 ladder, unchanged) fitted on B's 10 KB alone, and on A's 40 KB followed by B's 10 KB; scored on B's held-out.
import sys, time, json
import numpy as np
R = "/home/user/digitalbaby"
sys.path.insert(0, R + "/minifly_v148_pretrain_probe_dev_20261025")
import ladder
import importlib.util
_s = importlib.util.spec_from_file_location("v155_corpus", R + "/minifly_v155_keys_dev_v1_20261026/corpus.py")
corpus = importlib.util.module_from_spec(_s); sys.modules["v155_corpus"] = corpus; _s.loader.exec_module(corpus)
GUT = b"End of Project Gutenberg"
def splits(text, role, cut=False):
    recs = corpus.records(corpus.load(text))
    if cut:
        c = next((i for i, r in enumerate(recs) if r.startswith(GUT)), len(recs)); recs = recs[:c]
    return corpus.split(recs, *corpus.SIZES[role])
t0 = time.process_time()
out = {}
for A, B, cut in (("thursday", "edgeworth", False), ("emma", "ball", True)):
    sa, sb = splits(A, "A"), splits(B, "B", cut)
    held = sb["held"]; n = sum(len(r) + 1 for r in held)
    for k in (2, 3, 4, 5):
        for name, train in (("B_only", sb["train"]), ("A_then_B", sa["train"] + sb["train"])):
            m = ladder.PPMC(k).fit(train)
            tot, _ = ladder.bits(m, held)          # ladder.bits returns (total bits, symbols)
            b = tot / n
            out[f"{A}->{B}|ppmc{k}|{name}"] = round(b, 4)
            print(A, B, k, name, round(b, 4), flush=True)
print(json.dumps(out, indent=1)); print("cpu_s", round(time.process_time() - t0, 1))
