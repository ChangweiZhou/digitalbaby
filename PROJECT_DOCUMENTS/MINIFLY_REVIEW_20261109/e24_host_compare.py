#!/usr/bin/env python3
"""Why E24's host check stopped: one backprop birth (bank2, embedding decay off, answer targets) from three sources.

Usage:   python3 -I e24_host_compare.py [DEFAULT_REF] [E24_REF]
         defaults: origin/codex/minifly-consolidated-20261023 and origin/claude/charming-clarke-u9g850

For seed 73129 it sets side by side
  the pinned original         exp45/e17_dev_bank2_nbd_73129              born on the owner's machine, 9 November (commit fad2cec7)
  E24's host-check rerun      exp45/archive_e24_host/cloud_20261111/...  the harness itself on a cloud host, 11 November
  E20-B's G0 birth            exp45/e20b/e20_dev_G0_73129                E20-B's own runner on another cloud host, 11 November
and prints the hard mean (add89, sub89 at k = 12, whole answers) and the training loss at each checkpoint, then compares the three
weight files tensor by tensor. For the other two seeds it prints the final hard mean of the original and of E20-B's G0 birth.

Read-only: every file is read with `git show`; nothing is checked out, imported or trained. Standard library only.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_pt_tensors import compare, tensors          # noqa: E402  (the file beside this one)

DEFAULT = sys.argv[1] if len(sys.argv) > 1 else "origin/codex/minifly-consolidated-20261023"
E24 = sys.argv[2] if len(sys.argv) > 2 else "origin/claude/charming-clarke-u9g850"
BASE = "MOD97_LOCAL_RULE_20261106/exp45/"
SEEDS = (73129, 93683, 116089)


def show(ref, path):
    r = subprocess.run(["git", "show", f"{ref}:{BASE}{path}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def pct(d):
    return 100 * d["exact"] / d["n"]


def hard(row):
    return (pct(row["sealed"]["add89"]["12"]) + pct(row["sealed"]["sub89"]["12"])) / 2


def source(label, ref, stem):
    receipt, weights = show(ref, stem + ".json"), show(ref, stem + ".pt")
    return dict(label=label, receipt=json.loads(receipt) if receipt else None, weights=weights)


rows = [source("pinned original (owner's machine)", DEFAULT, "e17_dev_bank2_nbd_73129"),
        source("E24 host-check rerun (cloud)", E24, "archive_e24_host/cloud_20261111/e17_dev_bank2_nbd_73129"),
        source("E20-B G0 birth (cloud, own runner)", DEFAULT, "e20b/e20_dev_G0_73129")]
print("seed 73129, hard mean and training loss at each checkpoint")
for r in rows:
    if r["receipt"] is None:
        print(f"  {r['label']:36s} not found (is the branch fetched?)")
        continue
    curve = r["receipt"]["curve"]
    print(f"  {r['label']:36s} hard mean {[round(hard(x), 1) for x in curve]}  loss {[round(x['loss'], 4) for x in curve]}")

have = [r for r in rows if r["receipt"] is not None and r["weights"] is not None]
print("\nseed 73129, sealed scores and losses equal at every checkpoint, and weights tensor by tensor")
for i in range(len(have)):
    for j in range(i + 1, len(have)):
        a, b = have[i], have[j]
        ca, cb = a["receipt"]["curve"], b["receipt"]["curve"]
        same_scores = len(ca) == len(cb) and all(x["sealed"] == y["sealed"] for x, y in zip(ca, cb))
        same_loss = len(ca) == len(cb) and all(x["loss"] == y["loss"] for x, y in zip(ca, cb))
        common, equal, only_a, only_b = compare(tensors(a["weights"]), tensors(b["weights"]))
        print(f"  {a['label']}  vs  {b['label']}:\n      scores equal {same_scores}; losses equal {same_loss}; "
              f"tensors byte-equal {len(equal)} of {len(common)}" + (f"; unmatched names {only_a + only_b}" if only_a or only_b else ""))

print("\nfinal hard mean, the pinned originals against E20-B's G0 births (the same command on a cloud host)")
for s in SEEDS:
    a, b = show(DEFAULT, f"e17_dev_bank2_nbd_{s}.json"), show(DEFAULT, f"e20b/e20_dev_G0_{s}.json")
    if a is None or b is None:
        print(f"  {s}: not found")
        continue
    ha, hb = hard(json.loads(a)["curve"][-1]), hard(json.loads(b)["curve"][-1])
    print(f"  {s}: original {ha:.1f}; E20-B G0 birth {hb:.1f}; cloud minus original {hb - ha:+.1f}")
