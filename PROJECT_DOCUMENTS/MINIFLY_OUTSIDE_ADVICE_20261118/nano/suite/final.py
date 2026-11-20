#!/usr/bin/env python3
"""The last table of nano/suite: systems by streams, with the references.   python3 final.py > report_final.txt"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); ST = ("shakespeare", "book", "py", "c", "table", "xml")
L = lambda n: json.load(open(os.path.join(HERE, "res", n + ".json")))
ROWS = [("base: runs of 0 to 6 bytes, first cell", "base", 1), ("+ second cell (parts by kind and byte)", "gate1", 2), ("+ one recalling sender", "recall1", 2), ("+ the rule for the step", "rule2_eta0.03", 2), ("+ last-byte senders", "last", 2),
        ("Z: with three recalling senders", "Z", 2), ("ZA: Z with trusts by age", "Zage", 2), ("ZA, first cell", "Zage", 1), ("ZP: Z with since and pool cells", "ZP", 2), ("ZPA: ZP with trusts by age", "ZPA", 2), ("ZPA, first cell", "ZPA", 1),
        ("base with classes (nano/merge's program)", "merge_classes", 1), ("base with made kinds F (nano/merge's program)", "merge_F", 1)]
f = lambda v: f"{v:.4f}"
for m, title in (("going_on", "Everything going on (each byte of the last tenth answered before it is read; nats a byte)"), ("frozen", "Nothing changing (nats a byte)")):
    print(f"\n**{title}**\n\n| system | " + " | ".join(ST) + " |\n|---|" + "---|" * 6)
    for name, tag, cell in ROWS: print(f"| {name} | " + " | ".join(f(L(f"{s}__{tag}")[m][f"cell{cell}"]) for s in ST) + " |")
    r = {s: L(f"ref__{s}") for s in ST}
    if m == "going_on":
        for c in ("xz", "bzip2", "gzip"): print(f"| {c} at its strongest setting | " + " | ".join(f(r[s]["compressors"][c]["nats_a_byte_of_the_last_tenth"]) for s in ST) + " |")
    else: print("| best Kneser-Ney table of the last k bytes (k) | " + " | ".join((lambda k: f(r[s]["tables"][k]["kneser_ney_nats"]) + f" ({k})")(min(r[s]["tables"], key=lambda k: r[s]["tables"][k]["kneser_ney_nats"])) for s in ST) + " |")
print("\n**Size and cost** (at the scoring with nothing changing: cells that have counted; numbers stored = counts above zero and parts used; CPU seconds for the whole run)\n\n| system | " + " | ".join(ST) + " |\n|---|" + "---|" * 6)
for name, tag, cell in ROWS:
    if "first cell" in name and tag != "base": continue
    print(f"| {name} | " + " | ".join((lambda d: f"{d['sizes_at_the_frozen_scoring']['cells']:,}; {d['sizes_at_the_frozen_scoring']['counts_above_zero'] + d['sizes_at_the_frozen_scoring'].get('parts_used', 0):,}; {d['cpu']:.0f}")(L(f"{s}__{tag}")) for s in ST) + " |")
print("\n**The best general system against the strongest compressor here, going on** (ZA: no more cells than the base; ZPA: three to twelve times the cells)\n\n| | " + " | ".join(ST) + " |\n|---|" + "---|" * 6)
xz = {s: L(f"ref__{s}")["compressors"]["xz"]["nats_a_byte_of_the_last_tenth"] for s in ST}
for name, tag in (("ZA less xz", "Zage"), ("ZPA less xz", "ZPA")): print(f"| {name} | " + " | ".join(f"{L(f'{s}__{tag}')['going_on']['cell2'] - xz[s]:+.4f}" for s in ST) + " |")
print("| the better of ZA and ZPA, as a share of xz | " + " | ".join(f"{100 * min(L(f'{s}__Zage')['going_on']['cell2'], L(f'{s}__ZPA')['going_on']['cell2']) / xz[s]:.0f}%" for s in ST) + " |")
