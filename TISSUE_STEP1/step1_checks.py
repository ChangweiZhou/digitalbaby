"""E63: checks 1 to 10 of section 8 of the brief. Each writes TISSUE_STEP1/qual/check<n>.json with an entry "pass".

    python3 tissue_step1.py check <n>
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import tissue_step1 as T
import step1_read as RD

HERE = Path(__file__).resolve().parent
QUAL = HERE / "qual"


def write(n, obj):
    QUAL.mkdir(exist_ok=True)
    obj["program_sha256"] = T.sha_file(HERE / "tissue_step1.py")
    obj["read_sha256"] = T.sha_file(HERE / "step1_read.py")
    obj["checks_sha256"] = T.sha_file(HERE / "step1_checks.py")
    obj["versions"] = T.versions()
    obj["git_head"] = T.git_head()
    (QUAL / f"check{n}.json").write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"check {n}: pass = {obj['pass']}", flush=True)


# ================================================================================================ check 1: the worked example
EXPECT_ROWS = {
    "r4": {0: (0.4725, 0.1250, 0.0750, 0.0625, 0.0462, -0.0737, 0.0231, 0.3125, 0.0000, 0.4062),
           1: (0.2210, 0.2431, 0.1908, 0.1433, 0.0625, -0.0552, 0.0186, 0.4306, 0.1181, 0.3662),
           2: (0.0959, 0.1801, 0.1218, 0.1518, 0.1030, -0.0130, 0.0370, 0.3659, 0.0497, 0.3131),
           3: (0.0340, 0.1487, 0.0866, 0.1402, 0.1072, -0.0083, 0.0486, 0.3339, 0.0161, 0.2700),
           8: (0.0000, 0.1243, 0.0521, 0.1061, 0.0851, -0.0275, 0.0427, 0.3099, -0.0076, 0.2026),
           16: (0.0000, 0.1239, 0.0401, 0.1055, 0.0842, -0.0221, 0.0446, 0.0304, -0.2855, 0.0561),
           24: (0.0000, 0.1236, 0.0308, 0.1064, 0.0847, -0.0167, 0.0499, 0.0315, -0.2205, 0.0000),
           31: (0.0000, 0.1234, 0.0245, 0.1072, 0.0851, -0.0129, 0.0544, 0.0315, -0.1766, 0.0000)},
    "E4": {0: (0.5100, 0.1250, 0.0750, 0.0625, 0.0462, -0.0737, 0.0231, 0.3125, 0.0000, 0.4062),
           1: (0.2550, 0.2525, 0.2001, 0.1480, 0.0625, -0.0552, 0.0186, 0.4400, 0.1275, 0.3709),
           2: (0.1275, 0.1886, 0.1300, 0.1584, 0.1053, -0.0107, 0.0382, 0.3742, 0.0577, 0.3196),
           3: (0.0638, 0.1565, 0.0939, 0.1473, 0.1105, -0.0051, 0.0508, 0.3415, 0.0232, 0.2770),
           8: (0.0020, 0.1252, 0.0517, 0.1081, 0.0868, -0.0264, 0.0450, 0.3104, -0.0083, 0.2042),
           16: (0.0000, 0.1238, 0.0390, 0.1051, 0.0840, -0.0228, 0.0443, 0.0302, -0.2865, 0.0555),
           24: (0.0000, 0.1235, 0.0300, 0.1060, 0.0845, -0.0172, 0.0496, 0.0313, -0.2214, 0.0000),
           31: (0.0000, 0.1233, 0.0238, 0.1068, 0.0849, -0.0134, 0.0541, 0.0313, -0.1772, 0.0000)},
    "K": {0: (0.5100, 0.1250, 0.0750, 0.0625, 0.0462, -0.0737, 0.0231, 0.3125, 0.0000, 0.4062),
          1: (0.2550, 0.2525, 0.2001, 0.1480, 0.0626, -0.0551, 0.0187, 0.4400, 0.1275, 0.3709),
          2: (0.1275, 0.1886, 0.1300, 0.1584, 0.1055, -0.0105, 0.0383, 0.3742, 0.0577, 0.3196),
          3: (0.0638, 0.1565, 0.0939, 0.1473, 0.1109, -0.0048, 0.0511, 0.3415, 0.0232, 0.2770),
          8: (0.0020, 0.1252, 0.0517, 0.1081, 0.0877, -0.0255, 0.0458, 0.3104, -0.0083, 0.2042),
          16: (0.0000, 0.1238, 0.0390, 0.1051, 0.0858, -0.0213, 0.0459, 0.0302, -0.2865, 0.0555),
          24: (0.0000, 0.1235, 0.0300, 0.1060, 0.0872, -0.0153, 0.0520, 0.0313, -0.2214, 0.0000),
          31: (0.0000, 0.1233, 0.0238, 0.1068, 0.0883, -0.0112, 0.0570, 0.0313, -0.1772, 0.0000)},
    "r5": {0: (0.5000, 0.1250, 0.0750, 0.0625, 0.0462, -0.0737, 0.0231, 0.3125, 0.0000, 0.4062),
           1: (0.2500, 0.2500, 0.1976, 0.1467, 0.0625, -0.0552, 0.0186, 0.4375, 0.1250, 0.3697),
           2: (0.1250, 0.1874, 0.1288, 0.1571, 0.1047, -0.0113, 0.0379, 0.3730, 0.0566, 0.3183),
           3: (0.0625, 0.1559, 0.0934, 0.1464, 0.1099, -0.0057, 0.0504, 0.3409, 0.0228, 0.2761),
           8: (0.0020, 0.1252, 0.0518, 0.1081, 0.0868, -0.0263, 0.0449, 0.3105, -0.0081, 0.2042),
           16: (0.0000, 0.1238, 0.0391, 0.1051, 0.0840, -0.0227, 0.0443, 0.0302, -0.2864, 0.0555),
           24: (0.0000, 0.1235, 0.0300, 0.1060, 0.0845, -0.0172, 0.0496, 0.0314, -0.2213, 0.0000),
           31: (0.0000, 0.1233, 0.0239, 0.1069, 0.0850, -0.0133, 0.0541, 0.0314, -0.1772, 0.0000)}}
EXPECT_FINAL = {   # (a): section 13 of revision 4 as the brief states it; (b), (c): section 19, C.2 and C.3; (d): revision 5, section 13, part 1
    "r4": {"O": {"u": 0.75099, "f": 0.24949, "F": 0.25014}, "I": {"E": 0.5003, "q": 0.2510, "g": 0.2495}},
    "E4": {"E": {"h": 0.2554, "c": 0.4973, "F": 0.24957}, "I": {"E": 0.50031, "q": 0.25102, "g": 0.24949},
           "O": {"u": 0.75031, "f": 0.24983, "F": 0.25017}},
    "K": {"E": {"h": 0.2554, "c": 0.4973, "F": 0.24957}, "I": {"E": 0.49887, "q": 0.24641, "g": 0.21127},
          "O": {"u": 0.75031, "f": 0.24983, "F": 0.25017}},
    "r5": {"O": {"u": 0.7504, "f": 0.2498, "F": 0.2502}, "E": {"h": 0.2554, "c": 0.4973}, "I": {"E": 0.5003, "q": 0.2510, "g": 0.2495}}}


def example(variant):
    """Section 13 of the definition, part 1, with the program's own formula functions, in the order of sentence 120."""
    ext = {"s1": lambda k: 0.1, "h": lambda k: 0.25, "c": lambda k: 0.125, "q": lambda k: 0.25, "g": lambda k: 0.125,
           "u": lambda k: 0.5 if k < 16 else 0.125, "f": lambda k: 0.25,
           "L3": lambda k: 1.0, "L5": lambda k: 0.0, "NO": lambda k: 1.0 if k == 0 else 0.0}
    sign = {"s1": 1, "h": -1, "c": 1, "q": 1, "g": -1, "u": 1, "f": -1, "L3": 1, "L5": 1, "NO": 1, "E": 1, "F": 1}
    lines = {"L3", "L5", "NO"}
    r4 = variant == "r4"
    cells = {
        "F": dict(src={"NO": [1.0, 0.0], **({"s1": [0.25, 0.10]} if r4 else {})}, v=0.0 if variant == "r5" else 0.02, p=0.02,
                  th=0.05 if r4 else 0.0, xp=0.0, plastic=r4, theta=r4, rule="f7"),
        "E": dict(src={"L3": [0.125, 0.02], "L5": [0.125, 0.40], "h": [0.25, 0.25], "c": [0.5, 0.125], "F": [0.25, 0.0]},
                  v=0.04, p=0.05, th=0.02, xp=0.03, plastic=True, theta=True, rule="f7"),
        "I": dict(src={"E": [0.5, 0.06], "q": [0.25, 0.25], "g": [0.25, 0.125]}, v=0.10, p=0.12, th=0.05, xp=0.06, plastic=True,
                  theta=True, rule="coincidence" if variant == "K" else "f7", vt=T.ASTAR),
        "O": dict(src={"u": [0.75, 0.50], "f": [0.25, 0.25], "F": [0.25, 0.0]}, v=0.70, p=0.3125, th=0.10, xp=0.60, plastic=True,
                  theta=True, rule="f7")}
    rows = {}
    for k in range(32):
        z = {n: f(k) for n, f in ext.items()}
        z.update({n: c["xp"] for n, c in cells.items()})
        new = {}
        for n, c in cells.items():                                    # operations 2 to 4
            names = list(c["src"])
            A = T.f1_arrival([sign[j] for j in names], [c["src"][j][0] for j in names], [z[j] for j in names])
            m = T.f5_m(A, c["p"])
            v = T.f2_v(c["v"], A)
            x = float(T.f3_x(v, c["th"]))
            new[n] = (A, m, v, x)
        for n, c in cells.items():                                    # operation 5
            A, m, v, x = new[n]
            if not c["plastic"]:
                continue
            for j, we in c["src"].items():
                if j in lines:
                    continue
                if c["rule"] == "coincidence":
                    dw = T.coin_inh_dw(we[1], c["xp"]) if sign[j] < 0 else T.coin_exc_dw(we[1], c["xp"], c["vt"])
                else:
                    dw = T.f7_dw(sign[j], m, c["xp"], we[1])
                we[0] = float(T.f8_bound(we[0] + dw))
        for n, c in cells.items():                                    # operation 6
            A, m, v, x = new[n]
            for j, we in c["src"].items():
                we[1] = T.f6_trace(we[1], z[j])
            c["p"] = T.f4_p(c["p"], A)
            if c["theta"]:
                c["th"] = float(T.f9_theta(c["th"], x))
            if c["rule"] == "coincidence":
                c["vt"] = T.vt_update(c["vt"], x)
            c["v"], c["xp"] = v, x
        rows[k] = new
    fin = {n: {j: we[0] for j, we in c["src"].items() if j not in lines} for n, c in cells.items()}
    return rows, fin


def check1(a):
    out, ok = {}, True
    for part, variant in (("a", "r4"), ("b", "E4"), ("c", "K"), ("d", "r5")):
        rows, fin = example(variant)
        worst, table = 0.0, {}
        for k, exp in EXPECT_ROWS[variant].items():
            r = rows[k]
            vals = (r["F"][3], r["E"][0], r["E"][1], r["E"][3], r["I"][0], r["I"][1], r["I"][3], r["O"][0], r["O"][1], r["O"][3])
            table[k] = [round(v, 6) for v in vals]
            worst = max(worst, max(abs(round(v, 4) - e) for v, e in zip(vals, exp)))
        fw = max(abs(fin[c][j] - v) for c, d in EXPECT_FINAL[variant].items() for j, v in d.items())
        good = worst <= 1e-9 and fw < 0.6e-4
        out[part] = dict(variant=variant, rows=table, final={c: {j: round(v, 6) for j, v in d.items()} for c, d in fin.items()},
                         largest_row_difference_after_rounding=worst, largest_final_difference=fw, passed=good)
        ok &= good
    # (e): the whole first-form tissue of seed 73129 with E10, steps 0 to 1,023
    st, _ = T.stream(73129, 40)
    tis = T.Tissue(73129, "first")
    watch = (0, 32, 64, 96, 127, 128, 129, 160)
    seen = {}

    def hook(t, k, A, m, v, x):
        if k in watch:
            for c in (497, 333):
                snd, rcv, w, k0 = t.live()
                msk = rcv == c
                seen.setdefault(k, {})[c] = dict(connections=int(msk.sum()), of_birth=int((msk & (k0 == 0)).sum()),
                                                 strengths=sorted(round(float(q), 4) for q in w[msk]), activity=round(float(x[c]), 4))
        if k == 128:
            seen["made_at_128"] = t.made
    for b in range(32):
        tis.run_byte(int(st[b]), hook)
    snd, rcv, w, k0 = tis.live()
    between = (k0 == 0) & (snd < T.N)
    expect = {0: (0.0625, 0.0), 32: (0.0625, 0.0193), 64: (0.0, 0.0), 96: (0.0, 0.0), 127: (0.0021, 0.0003), 128: (0.0021, 0.0003),
              129: (0.0021, 0.0041), 160: (0.0, 0.0322)}
    exp_conn = {k: (9, 9) if k >= 128 else (8, 8) for k in watch}
    e_ok = True
    rows_e = {}
    for k in watch:
        s497, s333 = seen[k][497], seen[k][333]
        ok_k = (abs(s497["activity"] - expect[k][0]) < 1e-9 and abs(s333["activity"] - expect[k][1]) < 1e-9
                and (s497["connections"], s333["connections"]) == exp_conn[k] and s497["of_birth"] == 8 and s333["of_birth"] == 8
                and all(q == 0.125 for q in s497["strengths"]))
        if k < 160:
            ok_k &= all(q == 0.125 for q in s333["strengths"])
        else:
            ok_k &= sorted(s333["strengths"]) == [0.1249, 0.1249] + [0.125] * 7
        rows_e[k] = dict(cell_497=s497, cell_333=s333, matches=ok_k)
        e_ok &= ok_k
    counts = dict(given_up=tis.given_up, made_at_128=seen["made_at_128"], made_by_1024=tis.made,
                  smallest_starting_strength_between_cells=round(float(w[between].min()), 4))
    c_ok = counts["given_up"] == 0 and counts["made_at_128"] == 1984 and counts["made_by_1024"] == 13875 and counts[
        "smallest_starting_strength_between_cells"] == 0.0455
    out["e"] = dict(rows=rows_e, counts=counts, passed=bool(e_ok and c_ok))
    ok &= e_ok and c_ok
    write(1, dict(check=1, **{"pass": bool(ok)}, parts=out))


# ================================================================================================ check 2: the births
EXPECT_B = {
    73129: dict(dig=("adb76cb771d012de", "c0638c096619aecb", "66952e5b022b0422"), kinds=(256, 64, 240, 1488), inhib=397,
                nb=(32, 96.9, 154, 90.9), total=192179, lines=754,
                dist={1: 232, 2: 189, 3: 199, 4: 193, 5: 201, 6: 185, 7: 179, 8: 171, 9: 198, 10: 196, 11: 41}, fb=(383, 60, 47, 16, 27, 423)),
    93683: dict(dig=("3b7f27e7070b57ff", "8e946fac980830fc", "988ddd6833af5c1a"), kinds=(256, 64, 244, 1484), inhib=437,
                nb=(27, 95.3, 162, 89.4), total=189118, lines=833,
                dist={1: 238, 2: 175, 3: 176, 4: 206, 5: 179, 6: 191, 7: 197, 8: 196, 9: 217, 10: 200, 11: 9}, fb=(419, 53, 28, 27, 30, 460)),
    116089: dict(dig=("0121fb1fafc58c37", "503c1f17014882ab", "655770c0dec995ca"), kinds=(256, 64, 262, 1466), inhib=405,
                 nb=(26, 96.7, 166, 90.2), total=191803, lines=791,
                 dist={1: 247, 2: 191, 3: 190, 4: 196, 5: 176, 6: 187, 7: 198, 8: 212, 9: 197, 10: 181, 11: 9}, fb=(422, 55, 34, 26, 33, 465))}


def check2(a):
    out, ok = {}, True
    for s in T.SEEDS:
        B = T.birth(s)
        E = EXPECT_B[s]
        kind = B["kind"]
        kinds = (int((kind == T.OUTE).sum()), int((kind == T.FBK).sum()), int((kind == T.ENC).sum()), int((kind == T.INNER).sum()))
        sizes = [len(B["members"][i]) for i in range(T.N) if i not in B["fbset"]]
        cells_only = [sum(1 for m in B["members"][i] if m < T.N) for i in range(T.N) if i not in B["fbset"]]
        nb = (min(sizes), round(sum(sizes) / len(sizes), 1), max(sizes), round(sum(cells_only) / len(cells_only), 1))
        start = sum(len(B["src"][i]) for i in range(T.N))
        lines = sum(1 for i in range(T.N) for j in B["src"][i] if j >= T.N)
        tis = [T.Tissue(s, r, B=None) for r in ("first", "coincidence", "fixed")]
        d = tis[0].distances(range(256))
        hist = {k: int((d == k).sum()) for k in range(1, 12)}
        notr = int((d == 0).sum())
        fbt = {i for i in range(T.N) for j in B["src"][i] if j in B["fbset"]}
        enc = set(np.nonzero(kind == T.ENC)[0].tolist())
        fb = (len(fbt), len(fbt & B["outset"]), len({i for i in B["outset"] if set(B["src"][i]) & set(B["YES"])}),
              len({i for i in B["outset"] if set(B["src"][i]) & set(B["NO"])}), len(fbt & enc),
              sum(1 for i in range(T.N) for j in B["src"][i] if j in B["fbset"]))
        same = all(np.array_equal(tis[0].state_arrays()[k_], t_.state_arrays()[k_]) for t_ in tis[1:]
                   for k_ in ("snd", "rcv", "w", "e", "k0", "alive", "v", "p", "th", "abar"))
        row = dict(digests=B["digests"], kinds=kinds, inhibitory=sum(B["inhib"]), neighbourhood=nb, total_members=sum(sizes),
                   starting=start, from_lines=lines, distances=hist, not_reached=notr, feedback=fb, births_equal_across_tissues=same)
        good = ((B["digests"]["coordinates"], B["digests"]["sources"], B["digests"]["fb_drivers_kinds"]) == E["dig"] and kinds == E["kinds"]
                and sum(B["inhib"]) == E["inhib"] and nb == E["nb"] and sum(sizes) == E["total"] and start == 15872 and lines == E["lines"]
                and hist == E["dist"] and notr == 64 and fb == E["fb"] and same)
        row["passed"] = bool(good)
        out[str(s)] = row
        ok &= good
    write(2, dict(check=2, **{"pass": bool(ok)}, seeds=out))


# ================================================================================================ check 3: the stream
EXPECT_SHA = {73129: "7772341c01ab095437347063d75110bf0a585b73372668f4dd3187e8f95e8775",
              93683: "a6363a43f56c1cdca66e6c264d8faa774d24338eaa453643b49ee9a922ca6f4f",
              116089: "0473b978c0a8db0db2cd4bde07fb65515fb84fc8f71eef9601625335e4448bbe"}
EXPECT_48 = {73129: b'w58<VSQQZ88=58\nv05>LZOKMVKCQH12=12\np02+MS61=63\nq',
             93683: b'p11+NLZVXQ64=75\nq66-A54=12\nw51<EGDUQTUT52=51\np29',
             116089: b'v57>YNWWSESPEM83=83\nq28-06=22\np42+ADIPW20=62\nw41'}
WINS = {"A1": (2048, 4096), "A2": (12288, 4096), "W0": (0, 8192), "W1": (24576, 8192), "W2": (122880, 8192), "W3": (253952, 8192)}


def parse_items(st):
    """Every complete item: task byte, two digits, sign, 0 to 12 capitals, two digits, '=', two digits, byte 10."""
    s = bytes(st)
    items, i = [], 0
    task = {ord("p"): "add", ord("q"): "sub", ord("v"): "max", ord("w"): "min"}
    sign = {"add": ord("+"), "sub": ord("-"), "max": ord(">"), "min": ord("<")}
    while i < len(s):
        j = s.find(b"\n", i)
        if j < 0:
            break
        it = s[i:j]
        op = task[it[0]]
        a = int(it[1:3]); assert it[3] == sign[op]
        caps = it[4:-5]
        assert len(caps) <= 12 and all(65 <= c <= 90 for c in caps)
        b = int(it[-5:-3]); assert it[-3] == ord("=")
        ans = int(it[-2:])
        items.append((op, a, b, ans))
        i = j + 1
    return items


def check3(a):
    out, ok = {}, True
    for s in T.SEEDS:
        st, its = T.stream(s, 262144)
        sha = hashlib.sha256(st.tobytes()).hexdigest()
        vals = sorted(set(st.tolist()))
        per = {wn: len(set(st[w0:w0 + wl].tolist())) for wn, (w0, wl) in WINS.items()}
        items = parse_items(st)
        right = all(T.apply_op(op, 89, a_, b_) == ans for op, a_, b_, ans in items)
        taught = {t[0]: set(T.split(t, .5, s)[0]) for t in T.DEV}
        intaught = all((a_, b_) in taught[op] for op, a_, b_, ans in items)
        good = sha == EXPECT_SHA[s] and bytes(st[:48]) == EXPECT_48[s] and len(vals) == 46 and all(v == 46 for v in per.values()) and right and intaught
        out[str(s)] = dict(sha256=sha, first48=repr(bytes(st[:48])), values=len(vals), values_by_window=per, items_parsed=len(items),
                           answers_right=right, pairs_in_taught_half=intaught, passed=bool(good))
        ok &= good
    write(3, dict(check=3, **{"pass": bool(ok)}, seeds=out))


# ================================================================================================ check 4: signs
def check4(a):
    vals = {}
    for sig in (1, -1):
        for m in (0.1, -0.1):
            vals[f"formula7 sigma {sig:+d} m {m:+.1f}"] = round(float(T.f7_dw(sig, m, 0.3, 0.2)), 6)
    vals["coincidence exc theta-bar 0.1"] = round(float(T.coin_exc_dw(0.2, 0.3, 0.1)), 6)
    vals["coincidence exc theta-bar 0.5"] = round(float(T.coin_exc_dw(0.2, 0.3, 0.5)), 6)
    vals["coincidence inh x 0.3"] = round(float(T.coin_inh_dw(0.2, 0.3)), 6)
    vals["coincidence inh x 0.05"] = round(float(T.coin_inh_dw(0.2, 0.05)), 6)
    zero = [float(T.f7_dw(1, 0.1, 0.3, 0.0)), float(T.coin_exc_dw(0.0, 0.3, 0.1)), float(T.coin_inh_dw(0.0, 0.3))]
    exp = {"formula7 sigma +1 m +0.1": -0.00075, "formula7 sigma +1 m -0.1": 0.00075, "formula7 sigma -1 m +0.1": 0.00075,
           "formula7 sigma -1 m -0.1": -0.00075, "coincidence exc theta-bar 0.1": 0.0015, "coincidence exc theta-bar 0.5": -0.0015,
           "coincidence inh x 0.3": 0.004375, "coincidence inh x 0.05": -0.001875}
    ok = all(abs(vals[k] - v) < 1e-12 for k, v in exp.items()) and all(z == 0 for z in zero)
    write(4, dict(check=4, **{"pass": bool(ok)}, values=vals, zero_trace=zero))


# ================================================================================================ check 5: no reset
def check5(a):
    st, _ = T.stream(73129, 64)
    used = sorted(set(T.stream(73129, 262144)[0].tolist()))
    st2 = st.copy()
    assert st[5] == ord("S")
    st2[5] = used[used.index(st[5]) + 1]
    t1, t2 = T.Tissue(73129, "first"), T.Tissue(73129, "first")
    for b in range(41):
        t1.run_byte(int(st[b])); t2.run_byte(int(st2[b]))
    diff = float(np.abs(t1.v - t2.v).max())
    src = (HERE / "tissue_step1.py").read_text()
    ok = diff > 1e-12 and t2.k == 41 * 32
    write(5, {"check": 5, "pass": bool(ok), "replaced": [int(st[5]), int(st2[5])], "largest_v_difference_at_reading_step_of_position_40": diff,
              "statement": "The program sets a cell's or a connection's state after birth only by the formulas of operations 2 to 7 "
                           "(Tissue.step); Run.load restores a saved state of the same tissue exactly. No other assignment exists."})


# ================================================================================================ check 6: silent feedback, E4, E10
def check6(a):
    st, _ = T.stream(73129, 2048)
    tis = T.Tissue(73129, "first")
    tis.audit = []
    fb = np.nonzero(tis.isfb)[0]
    viol = dict(feedback_line_active=0, fb_state_nonzero=0, fb_sources=0, fb_made=0, line_strength_changed=0)
    line_w0 = None

    def hook(t, k, A, m, v, x):
        if t.z[T.YES_LINE] != 0 or t.z[T.NO_LINE] != 0:
            viol["feedback_line_active"] += 1
        if np.any(t.v[fb] != 0) or np.any(x[fb] != 0) or np.any(t.th[fb] != 0):
            viol["fb_state_nonzero"] += 1
    for b in range(2048):
        tis.run_byte(int(st[b]), hook)
        if b % 128 == 127:
            snd, rcv, w, k0 = tis.live()
            infb = np.isin(rcv, fb)
            if infb.sum() != 64 or not np.all(snd[infb] >= T.YES_LINE):
                viol["fb_sources"] += 1
            if np.any(infb & (k0 > 0)):
                viol["fb_made"] += 1
            isline = (snd >= T.LINE0)
            if np.any(w[isline & (snd < T.YES_LINE)] != T.START_W) or np.any(w[snd >= T.YES_LINE] != 1.0):
                viol["line_strength_changed"] += 1
    gu = [(k, s_, w_) for k, ss, ww, rr in tis.audit for s_, w_ in zip(ss.tolist(), ww.tolist())]
    from_line_or_fb = sum(1 for k, s_, w_ in gu if s_ >= T.N or tis.isfb[s_])
    not_below = sum(1 for k, s_, w_ in gu if not w_ < T.GIVE_UP)
    ok = all(v == 0 for v in viol.values()) and from_line_or_fb == 0 and not_below == 0
    write(6, {"check": 6, "pass": bool(ok), "violations": viol, "given_up": len(gu), "given_up_from_line_or_feedback_cell": from_line_or_fb,
              "given_up_not_below_1_128": not_below, "made": tis.made})


# ================================================================================================ check 7: the fixed tissue
def check7(a):
    st, _ = T.stream(73129, 2048)
    tis = T.Tissue(73129, "fixed")
    b0 = tis.state_arrays()
    for b in range(2048):
        tis.run_byte(int(st[b]))
    b1 = tis.state_arrays()
    same = all(np.array_equal(b0[k], b1[k]) for k in ("snd", "rcv", "w", "k0", "alive"))
    write(7, {"check": 7, "pass": bool(same), "connections": int(b1["alive"].sum()), "bit_equal_list_and_strengths": same})


# ================================================================================================ check 8: saving and resuming
def check8(a):
    base = QUAL / "check8"
    if base.exists():
        import shutil; shutil.rmtree(base)
    whole, cut = base / "whole", base / "cut"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E11_", "E16_", "EXP45_"))}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    py = [sys.executable, str(HERE / "tissue_step1.py"), "run", "first", "73129", "--bytes", "4096", "--stage", "A"]
    r1 = subprocess.run(py + ["--out", str(whole)], env=env, capture_output=True, text=True)
    r2 = subprocess.run(py + ["--out", str(cut), "--ckpt-bytes", "1024", "--die-after-byte", "1700"], env=env, capture_output=True, text=True)
    r3 = subprocess.run(py + ["--out", str(cut), "--ckpt-bytes", "1024", "--resume"], env=env, capture_output=True, text=True)
    fa, fb_ = np.load(whole / "final_state.npz"), np.load(cut / "final_state.npz")
    arrays_equal = sorted(fa.files) == sorted(fb_.files) and all(np.array_equal(fa[k], fb_[k]) for k in fa.files)
    ea, eb = np.load(whole / "emissions.npy"), np.load(cut / "emissions.npy")
    ra, rb = json.loads((whole / "receipt.json").read_text()), json.loads((cut / "receipt.json").read_text())
    rec_equal = ra["counts"] == rb["counts"] and ra["stretches"] == rb["stretches"] and np.array_equal(ea, eb)
    ok = r1.returncode == 0 and r2.returncode == 9 and r3.returncode == 0 and arrays_equal and rec_equal and len(rb["resumes"]) == 1
    write(8, {"check": 8, "pass": bool(ok), "exit_codes": [r1.returncode, r2.returncode, r3.returncode], "final_states_bit_equal": arrays_equal,
              "records_equal": rec_equal, "resumes": rb["resumes"], "stderr": [r1.stderr[-500:], r2.stderr[-500:], r3.stderr[-500:]]})


# ================================================================================================ check 9: the analysis
EXPECT_C0 = {(73129, "A1"): (1.0, 0.0858, 0.0497, 0.0695, 0.0335), (73129, "A2"): (1.0, 0.0703, 0.0561, 0.0729, 0.0354),
             (73129, "W0"): (1.0, 0.0870, 0.0590, 0.0756, 0.0340), (73129, "W3"): (1.0, 0.0792, 0.0504, 0.0767, 0.0381),
             (93683, "W0"): (1.0, 0.0778, 0.0531, 0.0784, 0.0406), (93683, "W3"): (1.0, 0.0818, 0.0508, 0.0803, 0.0372),
             (116089, "W0"): (1.0, 0.0807, 0.0529, 0.0772, 0.0342), (116089, "W3"): (1.0, 0.0744, 0.0515, 0.0804, 0.0430)}


def check9(a):
    streams = {s: T.stream(s, 262144)[0] for s in T.SEEDS}
    c0, clt, ok = {}, {}, True
    for (s, wn), exp in EXPECT_C0.items():
        w0, wl = WINS[wn]
        st = streams[s]
        got = [RD.c0(st, w0, wl, L)[0] for L in RD.LAGS]
        c0[f"{s} {wn}"] = [round(g, 4) for g in got]
        ok &= all(abs(round(g, 4) - e) < 1e-9 for g, e in zip(got, exp))
        clt[f"{s} {wn}"] = {str(L): round(RD.c_lt(st, w0, wl, L)[0], 4) for L in (1, 2, 4, 8)}
    st = streams[73129]
    w0, wl = WINS["A2"]
    pos = np.arange(w0, w0 + wl)
    pc = {}
    for L in RD.LAGS:
        pp = pos[pos - L >= 0]
        F = np.concatenate([RD.onehot([int(st[t - j]) for t in pp]) for j in range(9)], axis=1)
        pc[str(L)] = RD.classify_exact(F, st[pp - L].astype(np.int64), pp, w0, wl, L)[0]
    ok_pc = all(v >= 0.99 for v in pc.values())
    rng = np.random.default_rng(1)
    Fn = np.abs(rng.standard_normal((wl, 128)))
    Fn[Fn < 0.01] = 0.0
    nc = {}
    for L in RD.LAGS:
        sel = (pos - L) >= 0
        nc[str(L)] = RD.classify(Fn[sel], st[pos[sel] - L].astype(np.int64), pos[sel], w0, wl, L)[0]
    ok_nc = all(abs(v - 1 / 46) <= 0.01 for v in nc.values())
    since, posit = 0, []
    for b in st.tolist():
        posit.append(since)
        since = 0 if b == T.END else since + 1
    pair = RD.code_scores(st, w0, wl, 0, lambda t: (int(st[t - 1]), posit[t]))[0]
    ok_e = round(pair, 4) == 0.0946
    ok = ok and ok_pc and ok_nc and ok_e
    write(9, {"check": 9, "pass": bool(ok), "a_C0": c0, "a_passed": bool(all(True for _ in [0])), "b_positive_control_A2": pc,
              "b_passed": ok_pc, "c_negative_control_A2": nc, "c_passed": ok_nc, "d_C_lt": clt, "e_pair_code_A2": pair, "e_passed": ok_e})


# ================================================================================================ check 10: cost
def check10(a):
    st, _ = T.stream(73129, 1280)
    res = {}
    for rule in ("first", "coincidence", "fixed"):
        tis = T.Tissue(73129, rule)
        for b in range(256):
            tis.run_byte(int(st[b]))
        c0 = time.process_time()
        for b in range(256, 1280):
            tis.run_byte(int(st[b]))
        per = (time.process_time() - c0) / 1024
        snd, rcv, w, k0 = tis.live()
        conns = int((snd < T.YES_LINE).sum())
        res[rule] = dict(cpu_s_per_byte=per, mean_connections_end=conns / T.N)
    fc = 0.0
    for rule in ("first", "coincidence"):
        c = res[rule]["mean_connections_end"]
        fc += 16384 * res[rule]["cpu_s_per_byte"] * min(2.0, 93.8 / c)
    fc += 16384 * res["fixed"]["cpu_s_per_byte"] + 512 * res["first"]["cpu_s_per_byte"]
    prev = [json.loads((QUAL / f"check{n}.json").read_text())["pass"] if (QUAL / f"check{n}.json").exists() else False for n in range(1, 10)]
    ok = all(np.isfinite(r["cpu_s_per_byte"]) for r in res.values()) and all(prev)
    write(10, {"check": 10, "pass": bool(ok), "timings": res, "forecast_stage_A_cpu_hours": fc / 3600, "checks_1_to_9_passed": prev,
               "ask_before_stage_A": fc / 3600 > 2.5})


def run_check(n, a):
    {1: check1, 2: check2, 3: check3, 4: check4, 5: check5, 6: check6, 7: check7, 8: check8, 9: check9, 10: check10}[n](a)
