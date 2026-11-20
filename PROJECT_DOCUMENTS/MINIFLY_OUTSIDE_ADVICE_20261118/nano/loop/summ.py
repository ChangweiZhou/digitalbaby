#!/usr/bin/env python3
"""summ.py <prefix> [all]: the result files res/<prefix>*.json, sorted by the loss on the last tenth after the last reading (all: every reading)."""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); rows = []
for f in glob.glob(os.path.join(HERE, "res", sys.argv[1] + "*.json")):
    o = json.load(open(f))
    if o.get("rows"): rows.append(o)
for o in sorted(rows, key=lambda o: (o["args"].get("arm", "P"), o["rows"][-1]["nats"])):
    r = o["rows"][-1]; a = o["args"]
    print(f"{o['label']:34s} {a.get('arm','P'):2s} H={a.get('H','256'):5s} D={a.get('D','10'):2s} eta1={a.get('eta1','1'):4s} eta2={a.get('eta2','1/16'):4s} rscale={a.get('rscale','1'):4s} seed={a.get('seed','1'):6s} readings={r['passes']:2d}: "
          f"{r['nats']:.4f} nats {r['bits']:.4f} bits {r['right']:.2f}% | stream {r['stream_nats']:.4f} | cos {o.get('return_and_forward_cosine', float('nan')):.3f} | {o.get('cpu', r['cpu'])} s")
    if len(sys.argv) > 2: print("      " + " ".join(f"{q['passes']}:{q['nats']:.4f}/{q['right']:.1f}" for q in o["rows"]))
