#!/usr/bin/env python3
"""E22 (born and living under one rule) recomputed from its six life receipts and weight files.

Usage:   python3 -I e22_recompute.py [REF]      default REF: origin/codex/minifly-consolidated-20261023

Definitions are those of the registered validator (e22_validate.py): twelve distractors, whole answers, four operations weighted
equally, hard mean = (add97 + sub97) / 2, stage 1 = after block 1, stage 6 = end of life. This script shares no code with it.
It also checks each weights file against the hash in its receipt's completion record.

Read-only: every file is read with `git show`; nothing is checked out, imported or trained. Standard library only.
"""
import hashlib
import json
import subprocess
import sys

REF = sys.argv[1] if len(sys.argv) > 1 else "origin/codex/minifly-consolidated-20261023"
BASE = "MOD97_LOCAL_RULE_20261106/exp45/"
CELLS = (("rule-born", "e17_life_T_bank2_nbd_tall_bT_raw65536_e22"), ("backprop-born", "e17_life_T_bank2_nbd_tall_raw65536_e22"))


def show(name):
    r = subprocess.run(["git", "show", f"{REF}:{BASE}{name}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def pct(d):
    return 100 * d["exact"] / d["n"]


def fam(stages, i, mod, ops=("add", "sub", "max", "min")):
    return sum(pct(stages[i]["scores"][f"{o}{mod}"]["12"]) for o in ops) / len(ops)


largest = 0.0
for seed in (73129, 93683, 116089):
    row = {}
    for label, stem in CELLS:
        d = json.loads(show(f"{stem}_{seed}.json"))
        st, done = d["stages"], d.get("complete") or {}
        blob = show(done.get("weights", "missing"))
        row[label] = dict(b1=(fam(st, 1, 97), fam(st, 1, 97, ("add", "sub"))),
                          end=(fam(st, 6, 97), fam(st, 6, 97, ("add", "sub")), fam(st, 6, 73)), dev=fam(st, 6, 89), stages=len(st),
                          hash_ok=blob is not None and hashlib.sha256(blob).hexdigest() == done.get("weights_sha256"))
    a, b = row["rule-born"], row["backprop-born"]
    largest = max(largest, b["end"][0] - a["end"][0], b["end"][2] - a["end"][2])
    print(f"{seed}")
    print(f"  rule-born      after block 1: mod-97 {a['b1'][0]:.1f}, hard {a['b1'][1]:.1f} | end: mod-97 {a['end'][0]:.1f}, hard {a['end'][1]:.1f}, "
          f"mod-73 {a['end'][2]:.1f}, birth operations {a['dev']:.0f} | stages {a['stages']}, weights hash matches {a['hash_ok']}")
    print(f"  backprop-born  end: mod-97 {b['end'][0]:.1f}, hard {b['end'][1]:.1f}, mod-73 {b['end'][2]:.1f}, birth operations {b['dev']:.0f} "
          f"| stages {b['stages']}, weights hash matches {b['hash_ok']}")
    print(f"  partner minus rule-born: mod-97 {b['end'][0] - a['end'][0]:+.1f}, mod-73 {b['end'][2] - a['end'][2]:+.1f}")
print(f"largest gap, partner minus rule-born, on mod-97 or mod-73: {largest:.1f}")
