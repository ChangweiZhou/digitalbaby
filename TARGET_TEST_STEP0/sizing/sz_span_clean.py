"""Span roster, born blocks again with bytes the birth never used (brief OPUS_TASK_SIZING_SPAN_CLEAN_ALPHABET_20261113.md, commit 7287e071).

The copy fixture of the first span roster (sz_span.py, sz_crosscut.fixture_stream) with the 24 keys A-X, the 8 filler bytes, the cue '='
and the terminator '\\n' replaced by bytes that never occur in the E26 arm-B birth's training stream or its panels. Nothing else changes:
dose (235 x (D + 1) chunks fixed, 20,000 mixed), learner, starts, classes, last-tenth scoring, windows, repeats to 50%. sz_span, sz_crosscut
and sz_learner are imported unedited; sz_span.run is reused by pointing its module-level stream_for at this file's stream (the only
substitution), and the record's brief field is set to this brief.

    python3 sz_span_clean.py alphabet                     the byte census and the chosen bytes; writes runs/span_clean/ALPHABET.json
    python3 sz_span_clean.py admit                        stream checks and a cost forecast; writes runs/span_clean/admission.json
    python3 sz_span_clean.py run <block> <dist> <seed>    one run; writes runs/span_clean/<block>_<dist>_<seed>.json
    python3 sz_span_clean.py tables                       tables with the first roster's figures beside; writes runs/span_clean/CLEAN_TABLES.json

Roster (seeds 1104, 1105, 1106; births 73129, 93683, 116089 paired as in the first roster), 27 runs; block letters as in the first roster:
    C born      fixed 4, 12, 24, 40
    D born      mixed {4, 12, 24, 40}
    A random    fixed 24, 40
    P permuted  fixed 12, 24

Choices where the brief leaves room, fixed here before any fit:
  * Census: every byte value (0-255; the EOS target 256 and BOS 257 are not bytes) in the prompts and targets of the first 2,000 batches of
    exp11_cell.batches over the development tasks with the birth's seed and E.KMAX["M"], and of the birth's panels (E.panel per development
    task, ks 0, 4, 8, 12, 20, n 300), both built exactly as exp26_candidate.dev builds them; union over the three births.
  * "Printable" is ASCII 0x20-0x7E (space included). Free bytes are taken in ascending order: 24 keys, 8 filler, cue, terminator.
  * The stream is the first roster's stream with every byte relabelled: the same generators and seed strings draw the same key index, filler
    index and (mixed) distance, and new_key[i], new_fill[j] replace 65 + i and FILL[j]. Admission checks this byte for byte.
  * "Lost a learned copy" (for the report): a window of at least 100 instances at >= 90% right followed by a later window below 50%.
  * F8 and F10 compare repeats to 50%; a run that never reaches 50% counts as more repeats than one that does; two runs that never reach
    50% are not "fewer" either way; equal counts are not fewer.
"""
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_crosscut as C    # noqa: E402
import sz_learner as L     # noqa: E402
import sz_span as S        # noqa: E402

BRIEF = "OPUS_TASK_SIZING_SPAN_CLEAN_ALPHABET_20261113.md (7287e071)"
OUT = HERE / "runs" / "span_clean"
OLD = HERE / "runs" / "span"
MOD = HERE.parents[1] / "MOD97_LOCAL_RULE_20261106"
SEEDS = S.SEEDS
BIRTHS = tuple(S.BIRTH_OF[s] for s in SEEDS)
BLOCKS = dict(C=("born", (4, 12, 24, 40)), D=("born", ("M",)), A=("random", (24, 40)), P=("permuted", (12, 24)))
OLD_ALPHA = dict(keys=list(range(65, 65 + 24)), fill=list(C.FILL), cue=ord("="), term=ord("\n"))


# ---------------------------------------------------------------- alphabet
def census():
    sys.path.insert(0, str(MOD))
    import arch26
    arch26.install()
    import exp11_cell as E
    per = {}
    for seed in BIRTHS:
        tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
        bt = E.batches(tp, seed, E.KMAX["M"])
        sb = set()
        for _ in range(2000):
            for p, t in next(bt):
                sb.update(p); sb.update(t)
        pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
        sp = set()
        for d in pan.values():
            for its in d.values():
                for p, t in its:
                    sp.update(p); sp.update(t)
        per[seed] = dict(stream=sorted(x for x in sb if x < 256), panels=sorted(x for x in sp if x < 256),
                         non_byte_values=sorted(x for x in sb | sp if x >= 256))
    used = sorted(set().union(*(set(v["stream"]) | set(v["panels"]) for v in per.values())))
    return per, used


def alphabet():
    per, used = census()
    free = [b for b in range(0x20, 0x7F) if b not in used]
    note = None
    if len(free) < 34:
        free += [b for b in range(128, 256) if b not in used]
        note = "fewer than 34 printable bytes free; non-printable values from 128 upward added"
    keys, fill, cue, term = free[:24], free[24:32], free[32], free[33]
    rec = dict(brief=BRIEF, census=dict(per_birth={str(k): v for k, v in per.items()}, union=used, union_text=bytes(used).decode("latin-1"),
                                        batches_per_birth=2000, panels="E.panel per development task, ks 0,4,8,12,20, n 300"),
               printable="ASCII 0x20-0x7E", free_printable=[b for b in range(0x20, 0x7F) if b not in used], note=note,
               keys=keys, fill=fill, cue=cue, term=term,
               keys_text=bytes(keys).decode("latin-1"), fill_text=bytes(fill).decode("latin-1"), cue_text=chr(cue), term_text=chr(term),
               old=dict(OLD_ALPHA, keys_text="A-X", fill_text=C.FILL.decode(), cue_text="=", term_text="\\n"))
    assert len(set(keys + fill + [cue, term])) == 34 and not set(keys + fill + [cue, term]) & set(used)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ALPHABET.json").write_text(json.dumps(rec, indent=1))
    return rec


def load_alpha():
    a = json.loads((OUT / "ALPHABET.json").read_text())
    return dict(keys=a["keys"], fill=a["fill"], cue=a["cue"], term=a["term"])


# ---------------------------------------------------------------- streams
def gen(ds, n_bytes, seed, al, mixed):
    """The first roster's generators (sz_crosscut.fixture_stream for fixed D, sz_span.mixed_stream for mixed), bytes from `al`."""
    ds = tuple(sorted(ds)); label = ",".join(str(d) for d in ds)
    r = random.Random(f"sizing-crosscut|{label}|{seed}")
    rd = random.Random(f"sizing-span-D|{label}|{seed}") if mixed else None
    keys_, fill, cue, term = al["keys"], al["fill"], al["cue"], al["term"]
    out = bytearray(); keys = []
    while len(out) < n_bytes:
        D = rd.choice(ds) if mixed else ds[0]
        k = keys_[r.randrange(24)]
        keys.append((len(out), len(out) + D, k, D))
        out += bytes([k]) + bytes(r.choice(fill) for _ in range(D - 2)) + bytes([cue, k, term])
    return bytes(out), keys


def clean_stream_for(dist, seed, chunks, al=None):
    al = al or load_alpha()
    n = chunks * L.CHUNK
    data, keys = gen(S.M_SET if dist == "M" else (int(dist),), n, seed, al, dist == "M")
    data = data[:n]
    return data, [k for k in keys if k[1] < len(data)]


# ---------------------------------------------------------------- admission
def admit():
    al = load_alpha()
    used = set(json.loads((OUT / "ALPHABET.json").read_text())["census"]["union"])
    rec = dict(brief=BRIEF, host=dict(python=platform.python_version(), torch=T.__version__, machine=platform.machine(), platform=platform.platform()))
    # 1. with the old bytes this generator reproduces the first roster's streams byte for byte
    same = {}
    for dist in (4, 12, 24, 40, "M"):
        ch = 3000
        a, ka = clean_stream_for(dist, 1104, ch, OLD_ALPHA)
        b, kb = S.stream_for(dist, 1104, ch)
        same[str(dist)] = a == b and ka == kb
    # 2. with the new bytes: a relabelling of the old stream, every repeat equals its key at its D, no census byte occurs
    m = {**{o: n for o, n in zip(OLD_ALPHA["keys"], al["keys"])}, **{o: n for o, n in zip(OLD_ALPHA["fill"], al["fill"])},
         OLD_ALPHA["cue"]: al["cue"], OLD_ALPHA["term"]: al["term"]}
    relabel = {}; repeats = {}; clean = {}
    for dist in (4, 12, 24, 40, "M"):
        for seed in SEEDS:
            ch = S.chunks_for(dist)
            a, ka = clean_stream_for(dist, seed, ch, al)
            b, kb = S.stream_for(dist, seed, ch)
            key = f"{dist}_{seed}"
            relabel[key] = a == bytes(m[x] for x in b) and [(p, r, D) for p, r, _, D in ka] == [(p, r, D) for p, r, _, D in kb]
            repeats[key] = all(a[r] == k and a[p] == k and r - p == D and a[r - 1] == al["cue"] for p, r, k, D in ka)
            clean[key] = not (set(a) & used)
    rec["item1_old_bytes_reproduce_first_roster"] = dict(results=same, passed=all(same.values()))
    rec["item2_relabelling"] = dict(results=relabel, passed=all(relabel.values()))
    rec["item3_repeats_equal_keys"] = dict(results=repeats, passed=all(repeats.values()))
    rec["item4_no_census_byte_in_any_stream"] = dict(results=clean, passed=all(clean.values()))
    # 5. cost: a 300-chunk timing run (born, D = 12, seed 1104), not a roster run
    S.stream_for = clean_stream_for
    r = S.run("C", 12, 1104, chunks=300)
    pc = r["cpu"]["per_chunk"]
    total = sum(S.chunks_for(d) for _, ds in BLOCKS.values() for d in ds for _ in SEEDS)
    rec["cost_forecast"] = dict(runs=sum(len(ds) for _, ds in BLOCKS.values()) * 3, chunks=total, timing_run_chunks=300, cpu_per_chunk=pc,
                                cpu_hours=total * pc / 3600, wall_hours_at_three_processes=total * pc / 3600 / 3,
                                brief_estimate_cpu_hours=5.5)
    rec["admitted"] = all(rec[k]["passed"] for k in rec if k.startswith("item"))
    (OUT / "admission.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: (v["passed"] if isinstance(v, dict) and "passed" in v else v) for k, v in rec.items()}, indent=1))
    return rec


# ---------------------------------------------------------------- tables
def lost_copy(r):
    out = {}
    for D in r["classes"]:
        hi = False; ev = []
        for w in r["windows"]:
            v = w["by_D"].get(D)
            if not v:
                continue
            n = v["crossing"] + v["not_crossing"]
            if not n:
                continue
            acc = (v["crossing_right"] + v["not_crossing_right"]) / n
            if hi and acc < .5:
                ev.append(dict(window=w["window"], accuracy=round(acc, 3)))
            if n >= 100 and acc >= .9:
                hi = True
        if ev:
            out[D] = ev
    return out


def row(c):
    lt = c["last_tenth"]
    return dict(cls=c["cls"], last_tenth=f"{lt['right']}/{lt['n']}", accuracy=round(lt["accuracy"], 3),
                crossing=f"{lt['crossing_right']}/{lt['crossing']}", repeats_to_50=c["repeats_to_50"])


def fewer(a, b):
    """a needed fewer repeats to 50% than b."""
    if a is None:
        return False
    return b is None or a < b


def tables():
    new = {}; old = {}
    for f in sorted(OUT.glob("[ACDP]_*_*.json")):
        r = json.loads(f.read_text()); new[(r["block"], str(r["distance"]), r["seed"])] = r
    for (b, d, s) in list(new):
        p = OLD / f"{b}_{d}_{s}.json"
        if p.exists():
            old[(b, d, s)] = json.loads(p.read_text())
    out = dict(brief=BRIEF, blocks={}, lost={}, cpu={})
    for b, (start, dists) in BLOCKS.items():
        rows = {}
        for d in dists:
            for s in SEEDS:
                rn, ro = new.get((b, str(d), s)), old.get((b, str(d), s))
                if rn is None:
                    continue
                for D, c in rn["classes"].items():
                    rows.setdefault(D, {})[str(s)] = dict(new=row(c), old=row(ro["classes"][D]) if ro else None)
        spans = {}
        for tag in ("new", "old"):
            span = None; stop = None
            for D in sorted(rows, key=int):
                k = sum(v[tag] is not None and v[tag]["cls"] == "pass" for v in rows[D].values())
                if k >= 2 and stop is None:
                    span = int(D)
                elif stop is None:
                    stop = dict(D=int(D), seeds_passing=k)
            spans[tag] = dict(span=span, stops_at=stop, passes_by_D={D: sum(v[tag] is not None and v[tag]["cls"] == "pass" for v in rows[D].values()) for D in rows})
        out["blocks"][b] = dict(start=start, rows=rows, spans=spans)
        out["cpu"][b] = round(sum(r["cpu"]["training"] + r["cpu"]["generation"] for (bb, _, _), r in new.items() if bb == b), 1)
    for tag, res in (("new", new), ("old", old)):
        out["lost"][tag] = {f"{b}_{d}_{s}": lc for (b, d, s), r in sorted(res.items()) if (lc := lost_copy(r))}

    def g(b, D, s, tag, k="repeats_to_50"):
        v = out["blocks"][b]["rows"].get(str(D), {}).get(str(s))
        return None if v is None or v[tag] is None else v[tag][k]
    f7 = [g("C", 24, s, "new", "cls") for s in SEEDS]
    f8 = [dict(seed=s, clean=g("C", 4, s, "new"), old=g("C", 4, s, "old"), fewer=fewer(g("C", 4, s, "new"), g("C", 4, s, "old"))) for s in SEEDS]
    f9 = [dict(D=D, seed=s, clean=g("A", D, s, "new", "cls"), old=g("A", D, s, "old", "cls"), same=g("A", D, s, "new", "cls") == g("A", D, s, "old", "cls"))
          for D in (24, 40) for s in SEEDS]
    f10 = [dict(seed=s, born=g("C", 12, s, "new"), permuted=g("P", 12, s, "new"), born_fewer=fewer(g("C", 12, s, "new"), g("P", 12, s, "new"))) for s in SEEDS]
    out["forecasts"] = dict(
        F7=dict(text="Born start, clean bytes, D = 24: pass in three of three", classes=f7, held=f7.count("pass") == 3),
        F8=dict(text="Born start at D = 4 reaches 50% in fewer repeats with clean bytes than it did with the old ones, in all three seeds",
                seeds=f8, held=all(x["fewer"] for x in f8)),
        F9=dict(text="Random start at D = 24 and 40 gives the same class as with the old bytes in at least five of six runs",
                runs=f9, held=sum(x["same"] for x in f9) >= 5),
        F10=dict(text="Born start beats permuted born in repeats to 50% at D = 12 in at least two of three seeds", seeds=f10,
                 held=sum(x["born_fewer"] for x in f10) >= 2))
    out["runs"] = len(new); out["cpu_total"] = round(sum(out["cpu"].values()), 1)
    (OUT / "CLEAN_TABLES.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "alphabet":
        a = alphabet(); print("used:", a["census"]["union_text"]); print("keys:", repr(a["keys_text"]), "fill:", repr(a["fill_text"]),
                                                                          "cue:", repr(a["cue_text"]), "term:", repr(a["term_text"]), "note:", a["note"])
    elif cmd == "admit":
        sys.exit(0 if admit()["admitted"] else 1)
    elif cmd == "run":
        block, dist, seed = sys.argv[2], sys.argv[3], int(sys.argv[4])
        dist = dist if dist == "M" else int(dist)
        assert block in BLOCKS and dist in BLOCKS[block][1] and seed in SEEDS
        path = OUT / f"{block}_{dist}_{seed}.json"
        if path.exists():
            raise SystemExit(f"refused: {path.name} exists")
        al = load_alpha()
        S.stream_for = lambda d, s, ch: clean_stream_for(d, s, ch, al)     # the one substitution in sz_span.run
        r = S.run(block, dist, seed)
        r["brief"] = BRIEF; r["alphabet"] = al
        path.write_text(json.dumps(r, indent=1))
        print(block, dist, seed, {D: (c["cls"], round(c["last_tenth"]["accuracy"], 3), c["repeats_to_50"]) for D, c in r["classes"].items()},
              "cpu", r["cpu"]["training"], flush=True)
    elif cmd == "tables":
        t = tables(); print(json.dumps({b: v["spans"] for b, v in t["blocks"].items()}, indent=1)); print(json.dumps(t["forecasts"], indent=1))
    else:
        print(__doc__)
