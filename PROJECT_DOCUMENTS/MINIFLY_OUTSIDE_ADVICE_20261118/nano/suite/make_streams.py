#!/usr/bin/env python3
"""The test-bed of nano/suite (PLAN_1.md): six byte streams of different kinds, each a real text found on this machine or in this checkout.
    python3 make_streams.py            writes streams/<name>.bin (one byte a value, mapped) and manifests/<name>.json (how it was made)
    import make_streams as M; x, cut, S = M.load("py")
Nothing is downloaded. A stream is: the raw bytes (files joined in sorted order of path until the size is reached, then cut to the size); then
a map to at most 128 values: the values met in the first nine tenths, in ascending order of byte value; if there are more than 128 of them, the 127
most frequent there (ties to the lower value) and one value more for all others. A value met only in the last tenth also goes to the value for others.
The cut is int(0.9 * length), as nano/table/score.py has it."""
import hashlib, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..", "..", ".."); OUT = os.path.join(HERE, "streams"); MAN = os.path.join(HERE, "manifests")
NAMES = ("shakespeare", "book", "py", "c", "table", "xml")

def joined(top, suffix, size):
    paths = sorted(os.path.join(d, f) for d, _, fs in os.walk(top) for f in fs if f.endswith(suffix) and not os.path.islink(os.path.join(d, f)))
    parts, used, n = [], [], 0
    for p in paths:
        b = open(p, "rb").read(); parts.append(b); used.append((p, len(b))); n += len(b)
        if n >= size: break
    return b"".join(parts)[:size], used

def raw_of(name):
    if name == "shakespeare":
        p = os.path.join(HERE, "..", "data", "tinyshakespeare.txt"); return open(p, "rb").read(), [("nano/data/tinyshakespeare.txt", os.path.getsize(p))], "the programme's test text, whole"
    if name == "book":
        rel = "minifly_v155_keys_dev_v1_20261026/data/melville-moby_dick.txt"; b = open(os.path.join(ROOT, rel), "rb").read(); n = len(b); b = bytes(np.frombuffer(b, np.uint8)[np.frombuffer(b, np.uint8) != 13])
        return b, [(rel, n)], "one long English book of this checkout, whole, carriage returns (byte 13) removed as nano/table/gate.py does"
    if name == "py": b, u = joined("/usr/lib/python3.12", ".py", 1500000); return b, u, "Python source: the files *.py under /usr/lib/python3.12 in sorted order of path, joined, cut at 1,500,000 bytes"
    if name == "c": b, u = joined("/usr/include", ".h", 1500000); return b, u, "C headers: the files *.h under /usr/include in sorted order of path, joined, cut at 1,500,000 bytes"
    if name == "table":
        p = "/usr/share/perl/5.38.2/Unicode/Collate/allkeys.txt"; b = open(p, "rb").read(); return b, [(p, len(b))], "a data file with numbers in rows: the Unicode collation table shipped with perl, whole"
    if name == "xml":
        p = "/usr/share/mime/packages/freedesktop.org.xml"; b = open(p, "rb").read(); return b[:1500000], [(p, len(b))], "markup with short texts in many languages: the shared MIME database (XML), its first 1,500,000 bytes"
    raise KeyError(name)

def mapped(raw):
    r = np.frombuffer(raw, np.uint8); cut = int(0.9 * len(r)); cnt = np.bincount(r[:cut], minlength=256); seen = np.flatnonzero(cnt)
    if len(seen) <= 128 and np.all(np.isin(r[cut:], seen)): keep = seen; other = False
    else:
        other = True
        if len(seen) <= 127: keep = seen
        else: o = np.lexsort((np.arange(256), -cnt)); keep = np.sort(o[:127])
    tab = np.full(256, len(keep), np.int64); tab[keep] = np.arange(len(keep)); x = tab[r].astype(np.uint8); S = len(keep) + int(other)
    n_other = int((x == len(keep)).sum()) if other else 0; n_other_test = int((x[cut:] == len(keep)).sum()) if other else 0
    return x, cut, S, [int(v) for v in keep], other, n_other, n_other_test, int(len(np.unique(r)))

def make(name):
    raw, used, what = raw_of(name); x, cut, S, keep, other, n_other, n_other_test, distinct = mapped(raw); os.makedirs(OUT, exist_ok=True); os.makedirs(MAN, exist_ok=True)
    open(os.path.join(OUT, name + ".bin"), "wb").write(x.tobytes())
    man = dict(name=name, what=what, length=len(raw), cut=cut, last_tenth=len(raw) - cut, values=S, distinct_byte_values_in_the_raw_stream=distinct, has_a_value_for_others=other,
               bytes_mapped_to_the_value_for_others=n_other, of_them_in_the_last_tenth=n_other_test, kept_byte_values=keep, sha256_raw=hashlib.sha256(raw).hexdigest(), sha256_mapped=hashlib.sha256(x.tobytes()).hexdigest(),
               files_in_order=[[p, n] for p, n in used], files=len(used))
    json.dump(man, open(os.path.join(MAN, name + ".json"), "w"), indent=1); return man

def load(name):
    """The stream as values 0..S-1 (int32), the cut, S. Built if it is not there; its hash is checked against the manifest."""
    f = os.path.join(OUT, name + ".bin"); m = os.path.join(MAN, name + ".json")
    if not os.path.exists(f):
        old = json.load(open(m)) if os.path.exists(m) else None; new = make(name); assert old is None or old["sha256_mapped"] == new["sha256_mapped"], "the stream rebuilt on this machine is not the one of the manifest"
    man = json.load(open(m)); b = open(f, "rb").read(); assert hashlib.sha256(b).hexdigest() == man["sha256_mapped"]
    return np.frombuffer(b, np.uint8).astype(np.int32), man["cut"], man["values"]

if __name__ == "__main__":
    for name in NAMES:
        m = make(name); print(f"{name}: {m['length']} bytes, cut {m['cut']}, {m['values']} values ({m['distinct_byte_values_in_the_raw_stream']} distinct in the raw stream), {m['bytes_mapped_to_the_value_for_others']} bytes to the value for others ({m['of_them_in_the_last_tenth']} in the last tenth), {m['files']} files | {m['what']}")
    sys.path.insert(0, os.path.join(HERE, "..", "table")); import score
    x, cut, S = score.load(); y, c2, S2 = load("shakespeare"); assert S2 == S and c2 == cut and np.array_equal(x, y); print("shakespeare is nano/table/score.py's stream, value for value: PASSED")
