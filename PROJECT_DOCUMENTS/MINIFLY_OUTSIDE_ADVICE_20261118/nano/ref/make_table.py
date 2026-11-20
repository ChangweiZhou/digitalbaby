#!/usr/bin/env python3
"""Prints the report's table from the result files in res/ (nothing is computed here)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); R = lambda f: json.load(open(os.path.join(HERE, "res", f + ".json")))
have = lambda f: os.path.exists(os.path.join(HERE, "res", f + ".json"))
rows = []
def row(name, numbers, seen, s, cpu, note=""):
    rows.append(f"| {name} | {numbers} | {seen} | {s['nats']:.3f} | {s['bits']:.3f} | {s['right']:.1f} | {cpu} | {note} |")
n = lambda v: f"{v:,}"
t = R("tables"); T = t["tables"]
row("equal probabilities", "0", "0", dict(nats=4.1744, bits=6.0224, right=100 / 65), "0")
for k in range(9):
    r = T[str(k)]
    row(f"table of the last {k} bytes, Kneser-Ney", n(r["runs_seen"]) + " counts", n(t["stream"]), r["kn"], f"{t['cpu']:.0f} s for all nine" if k == 0 else "", f"unsmoothed: {r['unsmoothed_right']:.1f}% right")
for lab, name in (("small2k", "GPT 4 layers, 4 heads, 128 wide, 64 back, 2,000 steps"), ("small10k", "the same, 10,000 steps"),
                  ("large", "GPT 6 layers, 6 heads, 384 wide, 256 back, batch 16, 4,000 steps")):
    if not have("gpt_" + lab): continue
    o = R("gpt_" + lab); e = o["stream_exact"]; h = o["stream_half"]; est = o["nanogpt_estimate_final"]
    row(name, n(o["numbers_that_learn"]), n(o["bytes_seen"]), e, f"{o['cpu_learning']:.0f} s",
        f"nanoGPT's estimate {est['val']:.3f} (stream part {est['train']:.3f}); half-window reading {h['nats']:.3f}, {h['right']:.1f}%" + ("" if e["every"] == 1 else f"; exact figure on every {e['every']}th byte"))
for lab, name in (("d10_h256", "one layer, 10 bytes, 256 cells"), ("d10_h1024", "one layer, 10 bytes, 1,024 cells"), ("d10_h4096", "one layer, 10 bytes, 4,096 cells"),
                  ("d32_h1024", "one layer, 32 bytes, 1,024 cells"), ("d32_h4096", "one layer, 32 bytes, 4,096 cells")):
    if not have("onelayer_" + lab + "_whole"): continue
    o = R("onelayer_" + lab + "_whole"); note = ""
    if have("onelayer_" + lab):
        d = R("onelayer_" + lab); note = f"with the last 50,000 bytes of the stream kept out: {d['chosen']['stream']['nats']:.3f}, {d['chosen']['stream']['right']:.1f}% (development part {d['chosen']['development']['nats']:.3f})"
    row(name, n(o["numbers_that_learn"]), n(o["bytes_seen"]), o["last_pass"]["stream"], f"{o['cpu_learning']:.0f} s", note)
for lab, name in (("gru256", "GRU of 256 cells"), ("gru512", "GRU of 512 cells")):
    if not have("gru_" + lab + "_whole"): continue
    o = R("gru_" + lab + "_whole"); note = ""
    if have("gru_" + lab):
        d = R("gru_" + lab); note = f"with the last 50,000 bytes kept out: {d['chosen']['stream']['nats']:.3f}, {d['chosen']['stream']['right']:.1f}% (development part {d['chosen']['development']['nats']:.3f}, pass {d['chosen']['passes']})"
    row(name, n(o["numbers_that_learn"]), n(o["bytes_seen"]), o["last_pass"]["stream"], f"{o['cpu_learning']:.0f} s", note)
print("| Model | Numbers that learn | Bytes seen | Stream loss, nats | bits | Named right, % | CPU time (learning) | Beside it |\n|---|---|---|---|---|---|---|---|")
print("\n".join(rows))
