#!/usr/bin/env python3
"""A measurement, no plan of its own (it scores nothing): how the number of cells that count grows with the bytes read.
    python3 growth.py
For each prefix of the life A, B (the stream of the test text, then the three books) the number of different contexts met so far, by kind:
the table's cells (runs of 1 to 6 bytes) and the cells that step once a word (A, B, C of words.py). A cell exists for each context met."""
import json, os
import numpy as np
import life, table, words
HERE = os.path.dirname(os.path.abspath(__file__)); S = 65
x, segs, _ = life.load_all(); order = [(k, segs[k][0], segs[k][1]) for k in segs]          # the parts read, in order (each text's first nine tenths)
read = np.concatenate([np.arange(a if a else 6, c) for _, a, c in order]); n_all = len(read)
kinds = {f"run of {k}": table.ctx_codes(x, k) for k in range(1, 7)}; kinds.update({f"word cells {n}": w for n, w in zip("ABC", words.word_keys(x))})
marks = [62500, 125000, 250000, 500000, 1000000, n_all]; out = {"bytes_read": marks, "cells": {}, "met_once": {}}
for name, key in kinds.items():
    kv = key[read - 1]; out["cells"][name] = []; out["met_once"][name] = []
    for m in marks:
        u, c = np.unique(kv[:m], return_counts=True); out["cells"][name].append(int(len(u))); out["met_once"][name].append(int((c == 1).sum()))
tot = np.array([sum(out["cells"][k][i] for k in kinds) for i in range(len(marks))]); once = np.array([sum(out["met_once"][k][i] for k in kinds) for i in range(len(marks))])
out["total"] = tot.tolist(); out["total_met_once"] = once.tolist(); lg = np.log(np.array(marks, float))
out["exponent_total"] = float(np.polyfit(lg[:5], np.log(tot[:5]), 1)[0]); out["exponent_by_kind"] = {k: float(np.polyfit(lg[:5], np.log(np.array(out["cells"][k][:5], float)), 1)[0]) for k in kinds}
json.dump(out, open(os.path.join(HERE, "res", "growth.json"), "w"))
print("bytes read:", marks); print("cells in all:", tot.tolist(), "| per byte read:", np.round(tot / np.array(marks), 3).tolist()); print("of them met once:", np.round(once / tot, 3).tolist())
print("cells grow as (bytes read) to the power %.2f within the test text (62,500 to 1,000,000 bytes)" % out["exponent_total"]); print("by kind:", {k: round(v, 2) for k, v in out["exponent_by_kind"].items()})
