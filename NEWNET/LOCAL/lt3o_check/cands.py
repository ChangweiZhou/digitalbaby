"""Benchmark set: the unfinished round-2 straggler (costliest known corner) plus 12 candidates drawn at random (seed 0) from rounds 2-3."""
import json, random, sys
sys.path.insert(0, "..")
D = "../"
from search_pop import sample
rng = random.Random(2); jobs = [(f"p2_{i:03d}", sample(rng)) for i in range(200)]
done = {json.loads(l)["name"] for l in open(D + "round2_pop.jsonl")}
strag = [(n, c) for n, c in jobs if n not in done]
R = [json.loads(l) for f in ("round2_pop.jsonl", "round3_mech.jsonl") for l in open(D + f)]
R = [r for r in R if "slope" in r]
pick = random.Random(0).sample(R, 12)
CANDS = strag + [(r["name"], r["cand"]) for r in pick]
SECS = {r["name"]: r["secs"] for r in R}
if __name__ == "__main__":
    for n, c in CANDS: print(n, sum(p["n"] for p in c["pops"]), SECS.get(n))
