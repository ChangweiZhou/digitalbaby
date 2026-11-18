#!/usr/bin/env python3
"""The two runs of review 9.88, recomputed from their receipts: lives at zero replay, and lives with every next byte as the target.

Usage:   python3 -I zero_replay_every_byte_recompute.py [REF]      default REF: origin/codex/minifly-consolidated-20261023

Twelve distractors, whole answers unless marked final, four operations weighted equally in a family mean, hard mean = add and sub.
  Zero replay:  window learner on the drifting stream; the sign rule against backprop on the same pipeline (arm M).
  Every byte:   windowless two-bank learner; the sign-feedback sweep and backprop on the same pipeline, each beside its
                answer-target partner.

Read-only: every file is read with `git show`; nothing is checked out, imported or trained. Standard library only.
"""
import json
import statistics as st
import subprocess
import sys

REF = sys.argv[1] if len(sys.argv) > 1 else "origin/codex/minifly-consolidated-20261023"
BASE = "MOD97_LOCAL_RULE_20261106/exp45/"


def load(name):
    r = subprocess.run(["git", "show", f"{REF}:{BASE}{name}"], capture_output=True)
    return json.loads(r.stdout) if r.returncode == 0 else None


def pct(d):
    return 100 * d["exact"] / d["n"]


def fam(scores, mod, ops=("add", "sub", "max", "min")):
    return st.mean(pct(scores[f"{o}{mod}"]["12"]) for o in ops)


def plus_minus(scores, key="final"):
    c = scores["compose"]
    return 100 * (c["+-"][key] + c["-+"][key]) / (c["+-"]["n"] + c["-+"]["n"])


print("zero replay (window learner, drifting stream): end of life")
for seed in (400009, 600013, 300007):
    for arm, stem in (("sign rule", "e16_life_T_raw65536_drift_online_rs0_mac"), ("backprop", "e16_life_M_raw65536_drift_online_rs0_mac")):
        d = load(f"{stem}_{seed}.json")
        if d is None:
            print(f"  {seed} {arm}: receipt not found")
            continue
        end = d["stages"][-1]["scores"]
        best = max(plus_minus(x["scores"]) for x in d["stages"])
        print(f"  {seed} {arm:9s}: mod-97 {fam(end, 97):5.1f} | plus-minus pairs, final answer {plus_minus(end):4.1f} (best during the life "
              f"{best:4.1f}) | mod-73 {fam(end, 73):5.1f} | stages {len(d['stages'])}, completion record {bool(d.get('complete'))}")

print("\nevery next byte as the target (windowless two-bank learner): mod-97 mean after each block, and the end of life")
CELLS = (("sweep, every byte", "e17_life_T_bank2_nbd_tall_raw65536_lossall"), ("backprop, every byte", "e17_life_M_bank2_nbd_raw65536_lossall"),
         ("sweep, answer bytes", "e17_life_T_bank2_nbd_tall_raw65536"), ("backprop, answer bytes", "e17_life_M_bank2_nbd_raw65536"))
for seed in (73129, 93683, 116089):
    for arm, stem in CELLS:
        d = load(f"{stem}_{seed}.json")
        if d is None:
            print(f"  {seed} {arm}: receipt not found")
            continue
        end = d["stages"][-1]["scores"]
        blocks = [round(fam(x["scores"], 97)) for x in d["stages"][1:]]
        print(f"  {seed} {arm:22s}: by block {blocks} | end: mod-97 {fam(end, 97):5.1f}, hard {fam(end, 97, ('add', 'sub')):5.1f}, mod-73 {fam(end, 73):5.1f}")
