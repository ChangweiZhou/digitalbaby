#!/usr/bin/env python3
"""E20-B recomputed from its 24 receipts and weight files, independently of the worker's driver and summary script.

Usage:   python3 -I e20b_recompute.py [REF]      default REF: origin/codex/minifly-consolidated-20261023

Prints, for the four networks and three seeds:
  1. completion: a completion record, the registered step count, 6 curve rows per birth and 7 stage records per life, and the
     SHA-256 of the weights file against the hash the receipt records (the receipts' tensor hash is not recomputed here);
  2. the report's table at the end of life (whole answers, percent), plus the mod-97 family mean at 40 distractors, which the
     report's table does not carry;
  3. the selection rule of amendment 13;
  4. the three interrupted G0 lives against their reruns, stage by stage.

Read-only: every file is read with `git show`; nothing is checked out, imported or trained. Standard library only.
"""
import hashlib
import json
import statistics as st
import subprocess
import sys

REF = sys.argv[1] if len(sys.argv) > 1 else "origin/codex/minifly-consolidated-20261023"
BASE = "MOD97_LOCAL_RULE_20261106/exp45/e20b/"
SEEDS = (73129, 93683, 116089)
NETS = ("G0", "G1", "G2", "G3")
STEPS = {"dev": 24576, "life": 36864}


def show(name):
    r = subprocess.run(["git", "show", f"{REF}:{BASE}{name}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def load(name):
    raw = show(name)
    return json.loads(raw) if raw is not None else None


def pct(d):
    return 100 * d["exact"] / d["n"]


def fam(scores, mod, k="12"):
    return st.mean(pct(scores[f"{o}{mod}"][k]) for o in ("add", "sub", "max", "min"))


def field(receipt, key):
    """A birth keeps its completion fields at the top level (complete: true); a life keeps them inside `complete`."""
    done = receipt.get("complete")
    return done[key] if isinstance(done, dict) and key in done else receipt.get(key)


def steps_of(receipt):
    v = field(receipt, "optimizer_steps")
    return v[-1] if isinstance(v, list) else v


print("1. completion and weights hashes")
verified = 0
for kind in ("dev", "life"):
    for net in NETS:
        for s in SEEDS:
            name = f"e20_{kind}_{net}_{s}"
            d, blob = load(name + ".json"), show(name + ".pt")
            rows = len(d["curve"]) if kind == "dev" else len(d["stages"])
            ok = (bool(d.get("complete")) and steps_of(d) == STEPS[kind] and rows == (6 if kind == "dev" else 7)
                  and blob is not None and hashlib.sha256(blob).hexdigest() == field(d, "weights_sha256"))
            verified += ok
            if not ok:
                print(f"   NOT verified: {name}")
print(f"   verified {verified} of 24")

print("\n2. end of life (stage 6). pm = '+-' and '-+' pooled; F1 = mod 97, F3 = mod 73, dev = mod 89, family means at k = 12")
print("   net  seed    | birth hard | taught within |   +-    -+    ><    <> |    pm |   F1    F3   dev | F1 at k=40")
sel = {}
for net in NETS:
    for s in SEEDS:
        b, l = load(f"e20_dev_{net}_{s}.json"), load(f"e20_life_{net}_{s}.json")
        last = b["curve"][-1]["sealed"]
        hard = st.mean(pct(last[o]["12"]) for o in ("add89", "sub89"))
        sc = l["stages"][-1]["scores"]
        c = sc["compose"]
        pm = 100 * (c["+-"]["exact"] + c["-+"]["exact"]) / (c["+-"]["n"] + c["-+"]["n"])
        t = sc.get("taught")
        taught = pct(t) if "exact" in t else 100 * sum(v["exact"] for v in t.values()) / sum(v["n"] for v in t.values())
        print(f"   {net}  {s:7d} | {hard:10.1f} | {taught:6.1f} {pct(c['within']):6.1f} | {pct(c['+-']):5.1f} {pct(c['-+']):5.1f} "
              f"{pct(c['><']):5.1f} {pct(c['<>']):5.1f} | {pm:5.1f} | {fam(sc, 97):5.1f} {fam(sc, 73):5.1f} {fam(sc, 89):5.1f} | {fam(sc, 97, '40'):5.1f}")
        sel.setdefault(net, []).append((pct(c["within"]), pm))

print("\n3. selection rule (amendment 13): seeds with within >= 85 and plus-minus >= 40; tie broken by the three-seed plus-minus mean")
for net in NETS:
    q = sum(1 for w, p in sel[net] if w >= 85 and p >= 40)
    print(f"   {net}: {q} of 3 seeds; three-seed plus-minus mean {st.mean(p for _, p in sel[net]):.1f}; within mean {st.mean(w for w, _ in sel[net]):.1f}")

print("\n4. interrupted G0 lives against their reruns (scores and loss at every stage the interrupted run reached)")
for s in SEEDS:
    a, b = load(f"interrupted_20261111/e20_life_G0_{s}.json"), load(f"e20_life_G0_{s}.json")
    if a is None:
        print(f"   {s}: interrupted receipt not found")
        continue
    same = [a["stages"][i]["scores"] == b["stages"][i]["scores"] and a["stages"][i].get("loss") == b["stages"][i].get("loss")
            for i in range(len(a["stages"]))]
    print(f"   {s}: {len(same)} stages reached; identical to the rerun at each: {all(same)}")
