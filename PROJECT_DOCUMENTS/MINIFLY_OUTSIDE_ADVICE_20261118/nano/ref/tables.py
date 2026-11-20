#!/usr/bin/env python3
"""Tables of the last k bytes, k = 0..8, counted on the stream only (PLAN_1.md, item 2). Plain numpy.
    python3 tables.py            -> res/tables.json (probs(k) gives the probabilities themselves)
Three readings of the same counts:
  unsmoothed  the longest context of at most k bytes seen in the stream decides; share named right only
  kn          interpolated Kneser-Ney with three discounts for each order (Chen and Goodman's "modified" form)
  wb          interpolated Witten-Bell
Formulas in MEASURE.md."""
import json, os, sys, time
import numpy as np
import score as SC

KMAX = 8

def grams(stream, m):
    """Sorted distinct runs of m bytes in the stream as integers, and how often each occurs."""
    key = np.zeros(len(stream) - m + 1, np.int64)
    for d in range(m): key = key * SC.S + stream[d: len(stream) - m + 1 + d]
    return np.unique(key, return_counts=True)

def build(stream, kmax=KMAX):
    """raw[j]: counts of runs of j+1 bytes (context j, then the next byte). cont[j]: for each run of j+1 bytes,
    the number of different bytes seen in front of it (Kneser-Ney's count for the lower orders)."""
    raw = [grams(stream, j + 1) for j in range(kmax + 2)]
    cont = []
    for j in range(kmax + 1):
        longer = raw[j + 1][0] % (SC.S ** (j + 1))        # drop the first byte of each distinct run of j+2 bytes
        cont.append(np.unique(longer, return_counts=True))
    return raw[:kmax + 1], cont

def rows(table, ctxkey):
    """Counts of each of the 65 next bytes after each context: (n, 65)."""
    keys, counts = table
    q = ctxkey[:, None] * SC.S + np.arange(SC.S)[None, :]
    i = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return np.where(keys[i] == q, counts[i], 0).astype(np.float64)

def discounts(counts):
    """Chen and Goodman: Y = n1/(n1+2 n2); D1 = 1-2Y n2/n1; D2 = 2-3Y n3/n2; D3+ = 3-4Y n4/n3 (n_r: runs counted r times)."""
    n = [int((counts == r).sum()) for r in (1, 2, 3, 4)]
    if min(n) == 0: return np.array([0.0, 0.5, 0.5, 0.5]), n
    Y = n[0] / (n[0] + 2 * n[1])
    return np.array([0.0, 1 - 2 * Y * n[1] / n[0], 2 - 3 * Y * n[2] / n[1], 3 - 4 * Y * n[3] / n[2]]), n

def kn_step(C, D, lower):
    tot = C.sum(1, keepdims=True); d = D[np.minimum(C, 3).astype(np.int64)]
    seen = tot[:, 0] > 0; P = lower.copy()
    P[seen] = (C[seen] - d[seen]) / tot[seen] + d[seen].sum(1, keepdims=True) / tot[seen] * lower[seen]
    return P

def wb_step(C, lower):
    tot = C.sum(1, keepdims=True); kinds = (C > 0).sum(1, keepdims=True)
    seen = tot[:, 0] > 0; P = lower.copy()
    P[seen] = (C[seen] + kinds[seen] * lower[seen]) / (tot[seen] + kinds[seen])
    return P

def all_tables(stream, x, lo, hi, kmax=KMAX, keep=()):
    """Scores on the bytes lo..hi-1 of x for every k; tables counted on `stream`. keep: the k whose kn probabilities are returned."""
    raw, cont = build(stream, kmax); y = x[lo:hi]; n = len(y); out = {}; kept = {}
    uniform = np.full((n, SC.S), 1 / SC.S)
    q_kn = uniform; p_wb = uniform; best = np.zeros(n, np.int64); used = np.zeros(n, np.int64)
    for k in range(kmax + 1):
        ctx = np.zeros(n, np.int64)
        for d in range(k): ctx = ctx * SC.S + x[lo - k + d: hi - k + d]
        Craw = rows(raw[k], ctx); seen = Craw.sum(1) > 0
        best[seen] = Craw[seen].argmax(1); used[seen] = k   # unsmoothed: the longest context seen decides
        Dr, nr = discounts(raw[k][1]); Dc, nc = discounts(cont[k][1])
        p_kn = kn_step(Craw, Dr, q_kn)                      # the top order uses the counts themselves
        q_kn = kn_step(rows(cont[k], ctx), Dc, q_kn)        # the orders below use the numbers of different bytes in front
        p_wb = wb_step(Craw, p_wb)
        out[k] = dict(k=k, contexts_seen=int(len(np.unique(raw[k][0] // SC.S))), runs_seen=int(len(raw[k][0])),
                      unsmoothed_right=float(100 * (best == y).mean()), context_of_k_seen=float(100 * seen.mean()),
                      mean_context_used=float(used.mean()), kn=SC.score(p_kn, y), wb=SC.score(p_wb, y),
                      kn_discounts_top=Dr[1:].tolist(), kn_discounts_below=Dc[1:].tolist())
        if k in keep: kept[k] = p_kn
    return out, kept

def probs(k):
    """Kneser-Ney probabilities of the table of the last k bytes for the last tenth: (111540, 65)."""
    x, cut, _ = SC.load(); return all_tables(x[:cut], x, cut, len(x), kmax=k, keep=(k,))[1][k]

if __name__ == "__main__":
    t0 = time.process_time(); x, cut, _ = SC.load()
    if len(sys.argv) > 1 and sys.argv[1] == "without":      # PLAN_1.md, addition 1: counted without the stream's last 50,000 bytes, scored on the last tenth
        res, _ = all_tables(x[:cut - 50000], x, cut, len(x))
        json.dump(dict(counted_on=cut - 50000, tables=res), open(os.path.join(SC.HERE, "res", "tables_without_last50000.json"), "w"), indent=1)
        for k in range(KMAX + 1): print(f" {k} | {res[k]['unsmoothed_right']:6.2f} | {res[k]['kn']['nats']:.4f} {res[k]['kn']['right']:6.2f}")
        sys.exit()
    res, _ = all_tables(x[:cut], x, cut, len(x))
    dev = 50000                                             # the same tables counted without the stream's last 50,000 bytes, scored on them
    resdev, _ = all_tables(x[:cut - dev], x, cut - dev, cut)
    f = np.bincount(x[:cut], minlength=SC.S)
    out = dict(stream=cut, last_tenth=len(x) - cut, most_frequent_byte_right=float(100 * (x[cut:] == f.argmax()).mean()),
               tables=res, development_part=dict(bytes=dev, tables=resdev), cpu=round(time.process_time() - t0, 1))
    os.makedirs(os.path.join(SC.HERE, "res"), exist_ok=True)
    json.dump(out, open(os.path.join(SC.HERE, "res", "tables.json"), "w"), indent=1)
    print(f"stream {cut}, last tenth {len(x) - cut}; most frequent byte right {out['most_frequent_byte_right']:.2f}%; cpu {out['cpu']} s")
    print(" k | contexts seen | unsmoothed right % | KN nats  bits  right % | WB nats  bits  right % | development part: KN nats, right %")
    for k in range(KMAX + 1):
        r = res[k]; d = resdev[k]
        print(f" {k} | {r['contexts_seen']:>9} | {r['unsmoothed_right']:6.2f} | {r['kn']['nats']:.4f} {r['kn']['bits']:.4f} {r['kn']['right']:6.2f} |"
              f" {r['wb']['nats']:.4f} {r['wb']['bits']:.4f} {r['wb']['right']:6.2f} | {d['kn']['nats']:.4f} {d['kn']['right']:6.2f}")
