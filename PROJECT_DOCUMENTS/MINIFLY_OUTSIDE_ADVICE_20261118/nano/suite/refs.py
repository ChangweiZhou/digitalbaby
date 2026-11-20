#!/usr/bin/env python3
"""References that are themselves general methods, on the same split of each stream (PLAN_1.md).
    python3 refs.py <stream> ...          -> res/ref__<stream>.json
Compressors (xz -9e, bzip2 -9, gzip -9, on the mapped stream, one byte a value): the cost of the last tenth given the stream is the size of the whole
compressed less the size of the first nine tenths compressed, in nats a byte of the last tenth. A compressor goes on learning through the tenth.
Tables of the last k bytes (k = 3..6) with interpolated Kneser-Ney smoothing, counted on the first nine tenths only (nano/ref/tables.py, copied to ext/
unchanged; the number of values is set from outside, and a discount outside (0.05, r - 0.05) is moved to that bound): these change nothing during the tenth."""
import bz2, json, lzma, os, subprocess, sys, time, zlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "ext"))
import make_streams as MS

def sizes(b):
    run = lambda cmd: len(subprocess.run(cmd, input=b, stdout=subprocess.PIPE, check=True).stdout)
    return dict(xz=run(["xz", "-9e", "-c"]), bzip2=run(["bzip2", "-9", "-c"]), gzip=run(["gzip", "-9", "-c"]))

def one(name):
    x, cut, S = MS.load(name); b = x.astype(np.uint8).tobytes(); n = len(b) - cut; t0 = time.process_time(); whole = sizes(b); head = sizes(b[:cut]); out = dict(stream=name, S=S, length=len(b), cut=cut, compressors={})
    for c in whole: out["compressors"][c] = dict(whole_bytes=whole[c], first_nine_tenths_bytes=head[c], nats_a_byte_of_the_last_tenth=(whole[c] - head[c]) * 8 * np.log(2) / n, nats_a_byte_of_the_whole=whole[c] * 8 * np.log(2) / len(b))
    out["compressors"]["xz"]["reach"] = "the whole stream (dictionary of 64 MB at -9)"; out["compressors"]["bzip2"]["reach"] = "blocks of 900,000 bytes sorted apart: the tenth shares a block only with the bytes after the last block boundary before it"
    out["compressors"]["gzip"]["reach"] = "the last 32,768 bytes only"
    import score as SC, tables as TB
    SC.S = S; TB.SC.S = S; orig = TB.__dict__.setdefault("_discounts0", TB.discounts); clipped = []
    def disc(counts):                                             # the three discounts kept inside (0.05, r - 0.05): on rows of numbers the counts of counts can put one outside (0, r), which gives negative "probabilities"
        D, n = orig(counts); D2 = np.concatenate([[0.0], np.clip(D[1:], 0.05, np.array([1, 2, 3]) - 0.05)]); clipped.append(bool(np.any(D2 != D))); return D2, n
    TB.discounts = disc; r, _ = TB.all_tables(x[:cut].astype(np.int64), x.astype(np.int64), cut, len(x), kmax=6)
    out["tables"] = {str(k): dict(kneser_ney_nats=r[k]["kn"]["nats"], witten_bell_nats=r[k]["wb"]["nats"], runs_seen=r[k]["runs_seen"]) for k in (3, 4, 5, 6)}; out["discounts_clipped_in_orders"] = int(sum(clipped)); out["cpu_python"] = round(time.process_time() - t0, 1); return out

if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "res"), exist_ok=True)
    for name in sys.argv[1:]:
        o = one(name); json.dump(o, open(os.path.join(HERE, "res", f"ref__{name}.json"), "w"), indent=1)
        print(name, "| last tenth, nats a byte: ", {c: round(v["nats_a_byte_of_the_last_tenth"], 4) for c, v in o["compressors"].items()}, "| Kneser-Ney tables k=3..6:", [round(o["tables"][str(k)]["kneser_ney_nats"], 4) for k in (3, 4, 5, 6)], flush=True)
