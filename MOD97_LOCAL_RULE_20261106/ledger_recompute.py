"""Evidence ledger, job S1 (11 November 2026): recompute the numbers claimed in review sections 9.59-9.82 of
MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md and in section 2 of
PROJECT_DOCUMENTS/COORDINATOR_NOTE_WHAT_WAS_LEARNED_20261110.md, from the JSON receipts only.

Read-only. Python standard library only. Reads *.json under MOD97_LOCAL_RULE_20261106/ (exp45/ and E16F6_RUN_B/exp45/);
imports no model code, no torch, opens no .pt file, trains nothing, runs no life or birth, writes nothing.

Run:   python3 -I MOD97_LOCAL_RULE_20261106/ledger_recompute.py            (one line per claim, then the counts)
       python3 -I MOD97_LOCAL_RULE_20261106/ledger_recompute.py --summary  (counts only)

Output columns: claim-id | claimed | recomputed | match, where match is
  MATCH     every claimed number equals the recomputed one at the precision the claim is written in
            (a claimed "0.83" matches 0.825 <= x <= 0.835, so an exact tie such as 47.25 written 47.2 or 47.3 matches; a range "a-b"
            matches when the recomputed min and max are a and b at that precision; "about N" matches within 10% of N);
  WITHIN    a claimed range contains every recomputed value but its end points are not attained (a loose bound, not a contradiction);
  DIFFER    at least one number does not (the recomputed value is printed beside it; nothing is resolved here);
  NOT_RECOMPUTED   the number cannot be recomputed from JSON receipts; the reason is printed.

Definitions (as in the harness's own table scripts e16f7_gate.py, e16_audit_tables.py, e16m3_tables.py, summarize_exp16.py):
  operation score      exact answers at k = 12 distractors, per cent of the panel ('exact' / 'n')
  mod-97 / mod-73 mean mean of add, sub, max, min of that modulus at k = 12
  development ops      mean of add89, sub89, max89, min89 at k = 12 (never shown in life)
  held compositions    mean over the four held pairs +-, -+, ><, <> of final-answer accuracy
  hard pairs           mean of +- and -+ (final answer unless stated); hard-pair ratio = pooled learner / pooled reference
  whole worked answer  'exact' on the composition panels (intermediate ; final EOS)
  family gate          all three family ratios (mod-97, held compositions, mod-73) to the reference >= 0.80
"""
import json
import math
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.join(HERE, "exp45")
RUNB = os.path.join(HERE, "E16F6_RUN_B", "exp45")
F1 = ("add97", "sub97", "max97", "min97")
F3 = ("add73", "sub73", "max73", "min73")
DEV = ("add89", "sub89", "max89", "min89")
H = ("+-", "-+", "><", "<>")
S18 = (73129, 93683, 116089, 200003, 300007, 400009, 500011, 600013, 700027, 800011, 900007, 1000003,
       1100009, 1200007, 1300021, 1400017, 1500007, 1600033)
DEVSEEDS = (73129, 93683, 116089)
F6NEW = (500011, 600013, 700027, 800011, 900007, 1000003)
F7NEW = S18[12:]

_cache = {}


def J(name, base=X):
    p = os.path.join(base, name)
    if p not in _cache:
        _cache[p] = json.load(open(p)) if os.path.exists(p) else None
    return _cache[p]


def stages(name):
    d = J(name)
    if d is None:
        raise FileNotFoundError(name)
    return d["stages"]


def settings(name):
    s = [x["settings"] for x in stages(name) if "settings" in x]
    return s[-1] if s else {}


def last(name):
    return stages(name)[-1]["scores"]


def op(sc, n, k="12", key="exact"):
    return 100 * sc[n][k][key] / sc[n][k]["n"]


def fm(sc, fam, k="12"):
    return sum(op(sc, n, k) for n in fam) / len(fam)


def pr(sc, h, key="final"):
    c = sc["compose"][h]
    return 100 * c[key] / c["n"]


def comp(sc, key="final"):
    return sum(pr(sc, h, key) for h in H) / 4


def hard(sc, key="final"):
    return (pr(sc, "+-", key) + pr(sc, "-+", key)) / 2


def pooled(sc, pairs, key):
    return 100 * sum(sc["compose"][p][key] for p in pairs) / sum(sc["compose"][p]["n"] for p in pairs)


def fams(sc):
    return fm(sc, F1), comp(sc), fm(sc, F3)


def ratios(a, b):
    return [x / y for x, y in zip(fams(a), fams(b))]


def hard_ratio(a, b):
    return sum(a["compose"][h]["final"] for h in H[:2]) / sum(b["compose"][h]["final"] for h in H[:2])


def whole_ratio(a, b):
    return sum(a["compose"][h]["exact"] for h in H) / sum(b["compose"][h]["exact"] for h in H)


def birth_ops(seed):
    c = J(f"e11_dev_M_{seed}.json")["curve"][-1]["sealed"]
    return [op(c, n) for n in DEV]


def ranks(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    n = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return n / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


# ---------------------------------------------------------------- comparison helpers
def dec(s):
    s = s.strip().replace("−", "-").replace("+", "").replace(",", "")
    return Decimal(s)


def places(s):
    s = s.strip().replace("−", "-").replace(",", "")
    return len(s.split(".")[1]) if "." in s else 0


def rnd(x, p):
    q = Decimal(1).scaleb(-p)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def same(claimed, x):
    """claimed: a written number ('0.83', '79,218', '-5.8'); x: the recomputed float. Equal at the written precision
    (half-up rounding of the recomputed value)."""
    half = Decimal(5).scaleb(-places(claimed) - 1)
    return abs(Decimal(repr(float(x))) - dec(claimed)) <= half + Decimal("1e-9")


ROWS = []


def chk(cid, claimed_numbers, values, note=""):
    """claimed_numbers: list of written numbers; values: list of recomputed floats, same order."""
    assert len(claimed_numbers) == len(values), cid
    ok = all(same(c, v) for c, v in zip(claimed_numbers, values))
    rec = " / ".join(rnd(v, places(c) + 1) for c, v in zip(claimed_numbers, values))   # one decimal more than written
    ROWS.append((cid, " / ".join(claimed_numbers), rec + (f"  [{note}]" if note else ""), "MATCH" if ok else "DIFFER"))


def chk_range(cid, claimed_lo, claimed_hi, values, note=""):
    lo, hi = min(values), max(values)
    ok = same(claimed_lo, lo) and same(claimed_hi, hi)
    p = max(places(claimed_lo), places(claimed_hi))
    half = 0.5 * 10 ** -p + 1e-9
    inside = float(dec(claimed_lo)) - half <= lo and hi <= float(dec(claimed_hi)) + half
    pp = p + 1   # one decimal more than written
    ROWS.append((cid, f"{claimed_lo}–{claimed_hi}", f"{rnd(lo, pp)}–{rnd(hi, pp)}" + (f"  [{note}]" if note else ""),
                 "MATCH" if ok else ("WITHIN" if inside else "DIFFER")))


def chk_about(cid, claimed, values, note=""):
    """'about N': every recomputed value within 10% of N."""
    c = float(dec(claimed))
    ok = all(abs(v - c) <= 0.1 * abs(c) + 1e-9 for v in values)
    ROWS.append((cid, f"about {claimed}", f"{rnd(min(values), 1)}–{rnd(max(values), 1)}" + (f"  [{note}]" if note else ""), "MATCH" if ok else "DIFFER"))


def chk_fact(cid, claimed, recomputed, ok, note=""):
    ROWS.append((cid, claimed, recomputed + (f"  [{note}]" if note else ""), "MATCH" if ok else "DIFFER"))


def nr(cid, claimed, reason):
    ROWS.append((cid, claimed, "not recomputed: " + reason, "NOT_RECOMPUTED"))


def T_drift(s):
    return f"e16_life_T_raw65536_drift_online_{s}.json"


def D_drift(s):
    return f"e16_life_D_drift_{s}.json"


# ======================================================================= 9.59  E16-F6 run B
def s959():
    v = [x for s in (800011, 900007, 1000003) for x in birth_ops(s)]
    chk_range("9.59-a births 800011/900007/1000003 at k=12 (dev ops)", "89", "98", v)
    tw = [fams(last(D_drift(s))) for s in F6NEW]
    chk_fact("9.59-b twins at 90/85/90 or above (six F6 seeds)", ">= 90 / 85 / 90",
             "min " + " / ".join(rnd(min(t[i] for t in tw), 1) for i in range(3)),
             all(t[0] >= 90 and t[1] >= 85 and t[2] >= 90 for t in tw))
    chk("9.59-c 500011 twin compositions (drift)", ["88.4"], [comp(last(D_drift(500011)))])
    chk("9.59-c' 500011 twin compositions under stages", ["84.0"], [comp(last("e16_life_D_500011.json"))])
    r = {s: ratios(last(T_drift(s)), last(D_drift(s))) for s in F6NEW}
    npass = sum(min(x) >= .8 for x in r.values())
    chk_fact("9.59-d learners >= 0.80 on all three families", "5 of 6", f"{npass} of 6", npass == 5)
    chk_range("9.59-e composition ratios of the five passing seeds", "0.83", "0.89", [x[1] for s, x in r.items() if min(x) >= .8])
    chk("9.59-f 600013 compositions ratio, mod-97 ratio", ["0.705", "0.774"], [r[600013][1], r[600013][0]])
    a = last(T_drift(600013))
    chk_range("9.59-g 600013 +- and -+ (final)", "42", "48", [pr(a, "+-"), pr(a, "-+")])
    chk("9.59-g' 600013 max/min pairs (><, <> mean, final)", ["89"], [pooled(a, ["><", "<>"], "final")])
    chk_about("9.59-g'' 600013 add/sub mod 97 'near 50'", "50", [op(a, "add97"), op(a, "sub97")])
    chk("9.59-g''' 600013 twin 'untouched at 94' (twin mod-97 mean)", ["94"], [fm(last(D_drift(600013)), F1)])
    c2 = {s: fm(stages(T_drift(s))[2]["scores"], F1) for s in S18[:12]}
    fpm = {s: pr(last(T_drift(s)), "+-") for s in S18[:12]}
    chk("9.59-h Spearman ck2 mod-97 vs final +- (12 seeds)", ["0.217"], [spearman([c2[s] for s in S18[:12]], [fpm[s] for s in S18[:12]])])
    chk("9.59-h' same, nine re-keyed-panel seeds", ["0.450"], [spearman([c2[s] for s in S18[3:12]], [fpm[s] for s in S18[3:12]])])
    four = [s for s in S18[:12] if c2[s] > 80 and fpm[s] <= 65]
    chk_fact("9.59-i seeds above 80 at ck2 ending <= 65 on +-", "4", f"{len(four)} ({', '.join(map(str, four))})", len(four) == 4)
    m = [fm(x["scores"], F1) for x in stages(T_drift(600013))]
    chk("9.59-j 600013 mod-97 at ck3, ck2", ["47.2", "79.9"], [m[3], m[2]])
    mins = {s: min(fm(x["scores"], F1) for x in stages(T_drift(s))[1:]) for s in S18[:12]}
    chk("9.59-k life-minimum mod-97, failing seeds 600013, 400009", ["47.2", "62.0"], [mins[600013], mins[400009]])
    chk("9.59-k' lowest life-minimum among the ten passing seeds", ["66.7"], [min(v for s, v in mins.items() if s not in (400009, 600013))])
    g = sum(min(ratios(last(T_drift(s)), last(D_drift(s)))) >= .8 for s in S18[:12])
    chk_fact("9.59-l drift gate (family means) over twelve seeds", "10 of 12", f"{g} of 12", g == 10)
    # run B against run A (9.60 item 4): identical except timing
    names = [f"e11_dev_M_{s}.json" for s in (800011, 900007, 1000003)] + [T_drift(s) for s in F6NEW] + [D_drift(s) for s in F6NEW]

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "cpu"}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    same_n = sum(strip(J(n, RUNB)) == strip(J(n)) for n in names)
    byte_n = sum(J(n, RUNB) == J(n) for n in names)
    chk_fact("9.60(4) run B = run A in twelve receipts and three births", "bit-identical (15 files)",
             f"{same_n} of 15 equal in every field except 'cpu'; {byte_n} of 15 equal including 'cpu'", same_n == 15,
             note="CPU-time fields differ, as expected between two containers")
    nr("9.59-m pace 4.7 h against 5.3 forecast; within 3% of forecast CPU", "4.7 h / 5.3 h / 3%",
       "wall time and forecasts are in logs and the brief, not in JSON receipts")


# ======================================================================= 9.60  audit memo
def s960():
    nr("9.60-a embedding rows = birth x 9.9387e-5 (local arms, Phase 1)", "9.9387e-5", "needs the saved weights (.pt); this script opens no .pt file")
    chk("9.60-a' arithmetic: AdamW decay alone over 36,864 updates at 0.25 x 0.001", ["9.9387e-5".replace("e-5", "")],
        [(1 - 0.25 * 0.001) ** 36864 * 1e5], note="value x 1e5")
    chk("9.60-a'' per block of 6,144 updates", ["0.215"], [(1 - 0.25 * 0.001) ** 6144])
    z = settings("e17_life_T_bank2_nbd_raw65536_73129.json")["zero_grad_first_update"]
    zd = settings("e17_life_D_bank2_nbd_73129.json")["zero_grad_first_update"]
    chk_fact("9.60-b first update: local arm rows without gradient / rows in batch; backprop", "36 of 47; 2",
             f"{z['book_rows_in_batch_without_gradient']} of {z['book_rows_in_batch']}; {zd['book_rows_in_batch_without_gradient']} (seed 73129, Phase 2 receipts)",
             (z['book_rows_in_batch_without_gradient'], z['book_rows_in_batch'], zd['book_rows_in_batch_without_gradient']) == (36, 47, 2),
             note="the in-harness check is recorded in the Phase 2 receipts; Phase 1 receipts carry no such field")
    # the 21 raw-reservoir lives (gate_by_item_type logic, staged runs use *_oldpanels when present)
    tot = dict(n=0, fam=0, carry=0, whole=0, as97=0)
    cl, cr, ml, mr, ceil = [], [], [], [], []
    for stream in ("staged", "drift"):
        for s in S18[:12]:
            if stream == "staged":
                tn = f"e16_life_T_raw65536_{s}_oldpanels.json" if J(f"e16_life_T_raw65536_{s}_oldpanels.json") else f"e16_life_T_raw65536_{s}.json"
                dn = f"e16_life_D_{s}.json"
            else:
                tn, dn = T_drift(s), D_drift(s)
            if not J(tn) or not J(dn) or len(stages(tn)) != 7 or len(stages(dn)) != 7:
                continue
            a, b = last(tn), last(dn)
            fam = min(ratios(a, b))
            c1, c2 = pooled(a, ["+-", "-+"], "final"), pooled(b, ["+-", "-+"], "final")
            m1, m2 = pooled(a, ["><", "<>"], "final"), pooled(b, ["><", "<>"], "final")
            al = 100 * sum(a[n]["12"]["exact"] for n in ("add97", "sub97")) / sum(a[n]["12"]["n"] for n in ("add97", "sub97"))
            ar = 100 * sum(b[n]["12"]["exact"] for n in ("add97", "sub97")) / sum(b[n]["12"]["n"] for n in ("add97", "sub97"))
            sc0 = stages(tn)[0].get("shortcut_ceilings", {})
            ceil += [max(sc0[p].values()) for p in ("><", "<>") if p in sc0]
            cl.append(c1); cr.append(c2); ml.append(m1); mr.append(m2)
            tot["n"] += 1; tot["fam"] += fam >= .8; tot["carry"] += c1 / c2 >= .8
            tot["whole"] += whole_ratio(a, b) >= .8; tot["as97"] += al / ar >= .8
    chk_fact("9.60-c raw-reservoir lives: count; pass family / whole / add-sub97 / +- -+", "21; 18 / 14 / 15 / 4",
             f"{tot['n']}; {tot['fam']} / {tot['whole']} / {tot['as97']} / {tot['carry']}",
             (tot["n"], tot["fam"], tot["whole"], tot["as97"], tot["carry"]) == (21, 18, 14, 15, 4))
    chk_fact("9.60-c' not met on +- pairs alone", "17 of 21", f"{tot['n'] - tot['carry']} of {tot['n']}", tot["n"] - tot["carry"] == 17)
    chk("9.60-d comparison pairs (><,<>) learner / reference, mean of 21", ["89.5", "98.7"], [sum(ml) / len(ml), sum(mr) / len(mr)])
    chk("9.60-d' plus-minus pairs learner / reference, mean of 21", ["64.5", "86"], [sum(cl) / len(cl), sum(cr) / len(cr)])
    chk("9.60-d'' single-operator shortcut on ><,<> (mean of best ceiling)", ["69"], [100 * sum(ceil) / len(ceil)])
    nr("9.60-e exposure overlaps 40-52% / 47, 44, 21%", "40–52%; 47 / 44 / 21%", "from the 9 November memo's own script; no JSON receipt of these shares found")
    # 9.60 section 5 corrections
    v = []
    for s in DEVSEEDS:
        st = stages(f"e16_life_T_raw65536_lossall_{s}.json")
        v += [op(st[1]["scores"], "add97"), op(st[1]["scores"], "sub97")]
    chk("9.60-f E16-L add/sub after block 1 (three seeds)", ["46.8", "40.0", "29.8", "22.5", "21.5", "14.2"], v)
    b2 = max(op(stages(f"e16_life_T_raw65536_lossall_{s}.json")[2]["scores"], n) for s in DEVSEEDS for n in ("add97", "sub97"))
    b3 = max(op(stages(f"e16_life_T_raw65536_lossall_{s}.json")[3]["scores"], n) for s in DEVSEEDS for n in ("add97", "sub97"))
    chk_fact("9.60-f' E16-L add/sub under 12 after block 2, under 6 after block 3", "< 12; < 6", f"max {rnd(b2, 1)}; max {rnd(b3, 1)}", b2 < 12 and b3 < 6)
    n_, o_ = last("e16_life_T_raw65536_93683.json"), last("e16_life_T_raw65536_93683_oldpanels.json")
    chk("9.60-g E16-RP 93683 +- at block 6, run on new panels minus run on old panels", ["16.2"], [pr(n_, "+-") - pr(o_, "+-")])
    sw = [r for r in J("E16RP_SAME_WEIGHTS.json")["results"] if r["seed"] == 93683][0]
    chk("9.60-g' same weights, new minus old panel, +- (93683)", ["6.8"], [sw["diff_new_minus_old"]["+-"]["final"]])
    chk("9.60-g'' run-to-run part 'about 9 points'", ["9"], [pr(n_, "+-") - pr(o_, "+-") - sw["diff_new_minus_old"]["+-"]["final"]])
    v = [fm(x["scores"], F1) for s in DEVSEEDS for x in stages(f"e17_life_D_bank2_frz1_{s}.json")[1:]]
    chk_range("9.60-h frozen twin (E17 Phase 1b, D_bank2_frz1) mod-97 by block", "89.4", "95.0", v)
    rc = []
    for tag in ("raw16384", "raw32768", None, "raw131072"):
        names = [f"e16_life_T_{tag}_{s}.json" if tag else f"e16_life_T_raw65536_{s}_oldpanels.json" for s in DEVSEEDS]
        rc.append(sum(comp(last(n)) for n in names) / 3)
    chk("9.60-i E16-RC three-seed composition mean, 16,384 / 32,768 / 65,536 / 131,072", ["74.2", "76.9", "78.3", "79.8"], rc,
        note="65,536 = E16-R3 receipts (*_oldpanels), as in the E16-RC report")
    nr("9.60-j prediction scorecard 17 of 33; 1 of 12", "17 of 33; 1 of 12", "a count over prediction files, not receipts; outside this job's scope")


# ======================================================================= 9.61  E17 Phase 2
def e17(arm, s):
    return f"e17_life_{arm}_{s}.json"


def blockF1(arm, s, b):
    return fm(stages(e17(arm, s))[b]["scores"], F1)


def s961():
    U, Fz, XT, TW = "T_bank2_nbd_raw65536", "T_bank2_nbd_frz1_raw65536", "T_bank2_nbd_frz1_xfb_tup_raw65536", "D_bank2_nbd_frz1"
    chk("9.61-a P17.2.3 unfrozen sign rule F1 after block 1", ["2.8", "1.6", "1.5"], [blockF1(U, s, 1) for s in DEVSEEDS])
    chk("9.61-a' Phase 1 (with decay) F1 after block 1", ["1.5", "1.9", "1.9"], [blockF1("T_bank2_raw65536", s, 1) for s in DEVSEEDS])
    chk("9.61-b P17.2.4 frozen sign rule F1 after block 2, block 6", ["82.1", "86.9", "83.9", "59.4", "62.4", "61.2"],
        [blockF1(Fz, s, 2) for s in DEVSEEDS] + [blockF1(Fz, s, 6) for s in DEVSEEDS])
    chk("9.61-b' with decay (Phase 1b) after block 2, block 6", ["21", "15", "32", "20", "18", "17"],
        [blockF1("T_bank2_frz1_raw65536", s, 2) for s in DEVSEEDS] + [blockF1("T_bank2_frz1_raw65536", s, 6) for s in DEVSEEDS])
    chk("9.61-b'' 73129 misses 60 by", ["0.6"], [60 - blockF1(Fz, 73129, 6)])
    chk("9.61-c P17.2.5 arm after block 2, block 6", ["93.7", "93.6", "95.0", "87.5", "90.1", "91.9"],
        [blockF1(XT, s, 2) for s in DEVSEEDS] + [blockF1(XT, s, 6) for s in DEVSEEDS])
    chk("9.61-c' frozen twin after block 2, block 6", ["94.1", "95.0", "93.1", "90.2", "90.0", "91.5"],
        [blockF1(TW, s, 2) for s in DEVSEEDS] + [blockF1(TW, s, 6) for s in DEVSEEDS])
    chk("9.61-d P17.2.6 held pairs XT arm; frozen twin", ["13.4", "14.3", "15.0", "14.9", "17.8", "18.1"],
        [comp(last(e17(XT, s))) for s in DEVSEEDS] + [comp(last(e17(TW, s))) for s in DEVSEEDS])
    chk("9.61-d' XT arm +- ; -+", ["3.5", "2.7", "2.8", "3.2", "1.2", "1.3"],
        [pr(last(e17(XT, s)), "+-") for s in DEVSEEDS] + [pr(last(e17(XT, s)), "-+") for s in DEVSEEDS])
    z = [settings(e17(U, s))["zero_grad_first_update"] for s in DEVSEEDS]
    zd = [settings(e17("D_bank2_nbd", s))["zero_grad_first_update"] for s in DEVSEEDS]
    chk_fact("9.61-e zero-gradient rows on first update: unfrozen local arm; backprop", "36 of 47; 2",
             "; ".join(f"{a['book_rows_in_batch_without_gradient']}/{a['book_rows_in_batch']}" for a in z) + " | " + "; ".join(str(a["book_rows_in_batch_without_gradient"]) for a in zd),
             all(a["book_rows_in_batch_without_gradient"] == 36 and a["book_rows_in_batch"] == 47 for a in z) and all(a["book_rows_in_batch_without_gradient"] == 2 for a in zd))
    nr("9.61-f final/birth = 1.0000 for unreached rows; digit-row norm 4.1 -> 0.3, final/birth -0.40 to +1.56", "1.0000; 4.1 -> 0.3", "needs the saved weights (.pt)")
    chk_range("9.61-g unfrozen: development ops at birth", "93", "95", [fm(stages(e17(U, s))[0]["scores"], DEV) for s in DEVSEEDS])
    chk_range("9.61-g' unfrozen: development ops after block 1", "1", "2", [fm(stages(e17(U, s))[1]["scores"], DEV) for s in DEVSEEDS])
    chk_fact("9.61-g'' unfrozen: mod-97 never above 3 after birth", "< 3", "max " + rnd(max(blockF1(U, s, b) for s in DEVSEEDS for b in range(1, 7)), 1),
             max(blockF1(U, s, b) for s in DEVSEEDS for b in range(1, 7)) <= 3)
    chk_range("9.61-h frozen sign rule F1 through blocks 1-2", "82", "87", [blockF1(Fz, s, b) for s in DEVSEEDS for b in (1, 2)])
    chk_range("9.61-h' frozen sign rule low point in composition blocks (min of blocks 3-4)", "51", "54", [min(blockF1(Fz, s, 3), blockF1(Fz, s, 4)) for s in DEVSEEDS])
    chk_range("9.61-h'' frozen sign rule end", "59", "62", [blockF1(Fz, s, 6) for s in DEVSEEDS])
    chk_range("9.61-h''' frozen twin end", "90", "92", [blockF1(TW, s, 6) for s in DEVSEEDS])
    chk("9.61-i frozen sign rule add / sub at end, three seeds", ["41", "24", "46", "28", "46", "23"],
        [x for s in DEVSEEDS for x in (op(last(e17(Fz, s)), "add97"), op(last(e17(Fz, s)), "sub97"))])
    r = [ratios(last(e17(Fz, s)), last(e17(TW, s))) for s in DEVSEEDS]
    chk_range("9.61-j frozen sign rule ratio to frozen twin, mod-97", "0.66", "0.69", [x[0] for x in r])
    chk_range("9.61-j' ... mod-73", "0.77", "0.81", [x[2] for x in r])
    chk_range("9.61-j'' ... held compositions", "0.39", "0.52", [x[1] for x in r])
    d97 = [blockF1(XT, s, 6) - blockF1(TW, s, 6) for s in DEVSEEDS]
    chk_fact("9.61-k XT arm mod-97 within 3 of frozen twin, above in two", "|d| <= 3; 2 above", ", ".join(rnd(x, 1) for x in d97),
             all(abs(x) <= 3 for x in d97) and sum(x > 0 for x in d97) == 2)
    chk_range("9.61-k' XT arm mod-73 above twin by", "2", "3", [fm(last(e17(XT, s)), F3) - fm(last(e17(TW, s)), F3) for s in DEVSEEDS])
    chk_range("9.61-k'' XT arm add/sub at end", "84", "91", [op(last(e17(XT, s)), n) for s in DEVSEEDS for n in ("add97", "sub97")])
    chk_range("9.61-k''' XT arm held compositions ratio to twin", "0.81", "0.90", [comp(last(e17(XT, s))) / comp(last(e17(TW, s))) for s in DEVSEEDS])
    chk("9.61-l frozen twin +- ; -+", ["2.5", "10.7", "7.3", "2.5", "1.3", "2.0"],
        [pr(last(e17(TW, s)), "+-") for s in DEVSEEDS] + [pr(last(e17(TW, s)), "-+") for s in DEVSEEDS])
    for lab, arm, lo, hi in (("unfrozen sign rule", U, "7,475", "7,515"), ("frozen sign rule", Fz, "6,853", "6,923"),
                             ("XT arm", XT, "6,667", "6,808"), ("frozen twin", TW, "5,611", "5,738"), ("unfrozen twin", "D_bank2_nbd", "2,603", "2,624")):
        chk_range(f"9.61-m CPU-s per life, {lab}", lo, hi, [stages(e17(arm, s))[-1]["cpu"] for s in DEVSEEDS])
    nr("9.61-n wall 2 h 29 min; forecast; lock release 22:08", "2 h 29 min", "wall clock and lock are in logs, not JSON receipts")


# ======================================================================= 9.66  E17 Phase 3
def s966():
    XF, TU, FB, Fz, XT, TW = ("T_bank2_nbd_frz1_xfb_raw65536", "T_bank2_nbd_frz1_tup_raw65536", "T_bank2_nbd_frzb_raw65536",
                              "T_bank2_nbd_frz1_raw65536", "T_bank2_nbd_frz1_xfb_tup_raw65536", "D_bank2_nbd_frz1")
    chk("9.66-a P17.3.1 exact feedback alone F1 after block 6", ["67.9", "72.9", "72.8"], [blockF1(XF, s, 6) for s in DEVSEEDS])
    w8 = [blockF1(TW, s, 6) - blockF1(XF, s, 6) for s in DEVSEEDS]
    chk_fact("9.66-a' misses 'within 8 of frozen twin' in all three", "all three > 8 below", ", ".join(rnd(x, 1) for x in w8), all(x > 8 for x in w8))
    chk("9.66-b P17.3.2 trace alone F1 after block 6", ["80.1", "84.9", "84.9"], [blockF1(TU, s, 6) for s in DEVSEEDS])
    chk_range("9.66-b' trace alone above frozen sign rule by", "21", "24", [blockF1(TU, s, 6) - blockF1(Fz, s, 6) for s in DEVSEEDS])
    chk_range("9.66-b'' trace alone above exact-feedback-alone by '12'", "12", "12", [blockF1(TU, s, 6) - blockF1(XF, s, 6) for s in DEVSEEDS])
    chk("9.66-c P17.3.3 embeddings frozen: development ops after block 1", ["1.3", "1.5", "1.2"], [fm(stages(e17(FB, s))[1]["scores"], DEV) for s in DEVSEEDS])
    chk("9.66-c' F1 after block 1", ["1.8", "1.8", "1.1"], [blockF1(FB, s, 1) for s in DEVSEEDS])
    chk("9.66-d P17.3.4 held pairs, feedback; trace", ["8.8", "8.3", "8.8", "12.2", "13.2", "12.0"],
        [comp(last(e17(XF, s))) for s in DEVSEEDS] + [comp(last(e17(TU, s))) for s in DEVSEEDS])
    chk_range("9.66-d' hard pairs (+-, -+ each) in the three arms", "0", "4", [pr(last(e17(a, s)), h) for a in (XF, TU, FB) for s in DEVSEEDS for h in ("+-", "-+")])
    chk("9.66-e trace alone ratio to frozen twin, mod-97", ["0.89", "0.94", "0.93"], [blockF1(TU, s, 6) / blockF1(TW, s, 6) for s in DEVSEEDS])
    chk_range("9.66-e' ... mod-73", "0.97", "1.00", [fm(last(e17(TU, s)), F3) / fm(last(e17(TW, s)), F3) for s in DEVSEEDS])
    chk("9.66-e'' trace alone add / sub", ["78", "66", "84", "77", "83", "72"],
        [x for s in DEVSEEDS for x in (op(last(e17(TU, s)), "add97"), op(last(e17(TU, s)), "sub97"))])
    chk_range("9.66-e''' trace alone loss during composition blocks (block 2 - min(blocks 3, 4))", "10", "13",
              [blockF1(TU, s, 2) - min(blockF1(TU, s, 3), blockF1(TU, s, 4)) for s in DEVSEEDS])
    chk_range("9.66-f exact feedback alone adds to the sign rule (block 6)", "8", "12", [blockF1(XF, s, 6) - blockF1(Fz, s, 6) for s in DEVSEEDS])
    chk_fact("9.66-g embeddings frozen: mod-97 never above 2 after birth", "<= 2", "max " + rnd(max(blockF1(FB, s, b) for s in DEVSEEDS for b in range(1, 7)), 1),
             max(blockF1(FB, s, b) for s in DEVSEEDS for b in range(1, 7)) <= 2)
    chk_range("9.66-h 'every arm sits at 8-13 on the held pairs' (XF, TU, FB)", "8", "13", [comp(last(e17(a, s))) for a in (XF, TU, FB) for s in DEVSEEDS],
              note="XF and TU are 8.3-13.2; the embeddings-frozen arm FB is at 1.0-2.0")
    chk_range("9.66-h' frozen twin held pairs", "15", "18", [comp(last(e17(TW, s))) for s in DEVSEEDS])
    for lab, arm, lo, hi in (("exact feedback alone", XF, "4,840", "4,878"), ("trace alone", TU, "5,129", "5,152"), ("embeddings frozen", FB, "5,574", "5,602")):
        chk_range(f"9.66-i CPU-s per life, {lab}", lo, hi, [stages(e17(arm, s))[-1]["cpu"] for s in DEVSEEDS])
    nr("9.66-j wall 1 h 35 min, forecast 2 h 15", "1 h 35 min", "wall clock is in logs, not JSON receipts")


# ======================================================================= 9.62  E16-F7
def s962():
    v = [x for s in F7NEW for x in birth_ops(s)]
    chk("9.62-a P16F7.1 lowest birth op (sub89, 1400017)", ["86.3"], [min(v)])
    chk_fact("9.62-a' the lowest is sub89 of 1400017", "sub89, 1400017",
             str(min(((op(J(f"e11_dev_M_{s}.json")["curve"][-1]["sealed"], n), n, s) for s in F7NEW for n in DEV))[1:]),
             min(((op(J(f"e11_dev_M_{s}.json")["curve"][-1]["sealed"], n), n, s) for s in F7NEW for n in DEV))[1:] == ("sub89", 1400017))
    tw = [fams(last(D_drift(s))) for s in F7NEW]
    chk("9.62-b P16F7.2 lowest twin means (mod-97 / comps / mod-73)", ["93.7", "91.2", "92.2"], [min(t[i] for t in tw) for i in range(3)])
    r = {s: ratios(last(T_drift(s)), last(D_drift(s))) for s in F7NEW}
    np_ = sum(min(x) >= .8 for x in r.values())
    chk_fact("9.62-c P16F7.3 at >= 0.80 on all three families", "6 of 6", f"{np_} of 6", np_ == 6)
    chk("9.62-c' 1600033 passes by (mod-97 ratio - 0.80)", ["0.005"], [r[1600033][0] - 0.8])
    Sv = {s: min(fm(x["scores"], F1) for x in stages(T_drift(s))[2:5]) for s in S18}
    fpm = {s: pr(last(T_drift(s)), "+-") for s in S18}
    chk("9.62-d P16F7.4 Spearman S vs final +- over eighteen seeds", ["0.109"], [spearman([Sv[s] for s in S18], [fpm[s] for s in S18])])
    chk("9.62-d' Spearman among the six new seeds (exploratory)", ["-0.71"], [spearman([Sv[s] for s in F7NEW], [fpm[s] for s in F7NEW])])
    low = sorted(s for s in S18 if Sv[s] < 65)
    chk_fact("9.62-d'' seeds with S below 65 are the two known failures", "400009, 600013", ", ".join(map(str, low)), low == [400009, 600013])
    hp = {s: hard_ratio(last(T_drift(s)), last(D_drift(s))) >= .8 for s in S18}
    chk_fact("9.62-e P16F7.5 new seeds passing on +-/-+ ; family gate", "1 of 6; 6 of 6", f"{sum(hp[s] for s in F7NEW)} of 6; {np_} of 6", sum(hp[s] for s in F7NEW) == 1 and np_ == 6)
    fam18 = sum(min(ratios(last(T_drift(s)), last(D_drift(s)))) >= .8 for s in S18)
    chk_fact("9.62-f eighteen seeds: family means; hard pairs alone (mixed panel versions, see 9.75)", "16 of 18; 4 of 18",
             f"{fam18} of 18; {sum(hp.values())} of 18", fam18 == 16 and sum(hp.values()) == 4)
    chk("9.62-g mean +-/-+ learner; twin (18 seeds)", ["64.8", "87.5"], [sum(hard(last(T_drift(s))) for s in S18) / 18, sum(hard(last(D_drift(s))) for s in S18) / 18])
    chk("9.62-h 1400017 S; final +- (best of 18)", ["65.2", "79.0"], [Sv[1400017], fpm[1400017]])
    chk_fact("9.62-h' 1400017 has the best final +- of all eighteen", "best", "best" if max(fpm, key=fpm.get) == 1400017 else f"best is {max(fpm, key=fpm.get)}", max(fpm, key=fpm.get) == 1400017)
    chk("9.62-h'' S of 1200007, 1300021 ('near 82'); final +-", ["82", "82", "49.7", "51.5"], [Sv[1200007], Sv[1300021], fpm[1200007], fpm[1300021]])
    chk("9.62-i +- loss ck4 -> ck6, 1200007; 1300021", ["15", "30"],
        [pr(stages(T_drift(s))[4]["scores"], "+-") - pr(stages(T_drift(s))[6]["scores"], "+-") for s in (1200007, 1300021)])
    chk("9.62-i' 1200007 passes family gate with +- at; failing 400009 +-", ["49.7", "48.0"], [fpm[1200007], fpm[400009]],
        note="review writes 'level with the failing 400009' without a number; 48.0 is the receipt value")
    births = [J(f"e11_dev_M_{s}.json")["curve"][-1]["cpu"] for s in F7NEW]
    lrn = [stages(T_drift(s))[-1]["cpu"] for s in F7NEW]
    tws = [stages(D_drift(s))[-1]["cpu"] for s in F7NEW]
    chk("9.62-j total CPU-s, eighteen runs", ["79,218"], [sum(births) + sum(lrn) + sum(tws)])
    chk_range("9.62-j' births CPU-s", "3,051", "3,164", births)
    chk_range("9.62-j'' learners CPU-s", "5,329", "5,402", lrn)
    chk_range("9.62-j''' twins CPU-s", "4,643", "4,848", tws)
    nr("9.62-k 5 h 37 min wall; 4% over forecast; 4-core 2.10 GHz", "5 h 37 min", "wall time and machine are in logs; receipts carry no machine field")


# ======================================================================= 9.64  E16-M3
TAG = {0.0: "_rs0", 0.25: "_rs25", 0.5: "", 0.75: "_rs75", 0.875: "_rs88", 0.9375: "_rs94"}


def M3(arm, sh, s):
    return f"e16_life_{arm}_raw65536_drift_online{TAG[sh]}_{s}.json"


def s964():
    claimed = {400009: ["1.2", "43.7", "46.3", "73.2", "78.2", "84.7", "88.1", "89.1"],
               600013: ["1.0", "23.3", "44.8", "60.2", "71.9", "81.8", "86.7", "91.2"],
               300007: ["1.1", "59.2", "67.7", "75.7", "85.8"]}
    for s, c in claimed.items():
        vals = [hard(last(M3("T", sh, s))) for sh in (0.0, 0.25, 0.5, 0.75)]
        if s != 300007:
            vals += [hard(last(M3("M", sh, s))) for sh in (0.25, 0.5, 0.75)]
        vals += [hard(last(D_drift(s)))]
        chk(f"9.64-a share table, seed {s} (rule 0/.25/.5/.75{', backprop .25/.5/.75' if s != 300007 else ''}, twin)", c, vals)
    tw = {s: last(D_drift(s)) for s in (400009, 600013, 300007)}
    chk("9.64-b P16M3.1 share-0 mod-97 ratios", ["0.35", "0.32", "0.13"], [ratios(last(M3("T", 0.0, s)), tw[s])[0] for s in (400009, 600013, 300007)])
    chk("9.64-c P16M3.2 rule hard pairs 0.75 minus 0.5 (400009, 600013)", ["27", "15"], [hard(last(M3("T", .75, s))) - hard(last(M3("T", .5, s))) for s in (400009, 600013)])
    chk("9.64-d P16M3.3 backprop 0.5 minus 0.25 hard pairs (400009, 600013)", ["6", "10"], [hard(last(M3("M", .5, s))) - hard(last(M3("M", .25, s))) for s in (400009, 600013)])
    chk("9.64-e P16M3.4 300007 comp ratio at 0.25; hard 0.25 ; 0.75", ["0.796", "59.2", "75.7"],
        [ratios(last(M3("T", .25, 300007)), tw[300007])[1], hard(last(M3("T", .25, 300007))), hard(last(M3("T", .75, 300007)))])
    chk("9.64-f rule hard pairs gain 0.25 -> 0.75 (400009, 600013); backprop", ["29", "37", "10", "15"],
        [hard(last(M3("T", .75, s))) - hard(last(M3("T", .25, s))) for s in (400009, 600013)] + [hard(last(M3("M", .75, s))) - hard(last(M3("M", .25, s))) for s in (400009, 600013)])
    r4 = ratios(last(M3("T", .75, 400009)), tw[400009])
    chk("9.64-g 400009 at 0.75: family ratios; hard ratio", ["0.93", "0.87", "0.97", "0.82"], r4 + [hard_ratio(last(M3("T", .75, 400009)), tw[400009])])
    chk("9.64-g' 600013 at 0.75: comps ratio; hard ratio", ["0.796", "0.66"], [ratios(last(M3("T", .75, 600013)), tw[600013])[1], hard_ratio(last(M3("T", .75, 600013)), tw[600013])])
    chk("9.64-g'' 300007 at 0.75 hard ratio", ["0.88"], [hard_ratio(last(M3("T", .75, 300007)), tw[300007])])
    chk_range("9.64-h share 0 hard pairs at end ('about 1%')", "1", "1", [hard(last(M3("T", 0.0, s))) for s in tw])
    chk_range("9.64-h' share 0 peak hard pairs (max of ck3, ck4)", "59", "85", [max(hard(stages(M3("T", 0.0, s))[c]["scores"]) for c in (3, 4)) for s in tw])
    chk_range("9.64-h'' share 0 mod-97 ratio range", "0.13", "0.35", [ratios(last(M3("T", 0.0, s)), tw[s])[0] for s in tw])
    chk("9.64-i 400009 hard-pair peak at ck3, share 0.5 ; 0.75", ["57", "41"], [hard(stages(M3("T", .5, 400009))[3]["scores"]), hard(stages(M3("T", .75, 400009))[3]["scores"])])
    rule = [stages(M3("T", sh, s))[-1]["cpu"] for s in tw for sh in (0.0, 0.25, 0.75)]
    bp = [stages(M3("M", sh, s))[-1]["cpu"] for s in (400009, 600013) for sh in (0.25, 0.75)] + [stages(M3("M", .5, 600013))[-1]["cpu"]]
    chk_range("9.64-j rule lives CPU-s (this job's nine)", "5,132", "5,441", rule)
    chk_range("9.64-j' backprop lives CPU-s (this job's five)", "4,665", "4,733", bp)
    chk_fact("9.64-k fourteen drift lives in the job", "14", f"{len(rule) + len(bp)}", len(rule) + len(bp) == 14)
    nr("9.64-l 5.0 h wall; 3-9% over forecast", "5.0 h", "wall time is in logs, not JSON receipts")


# ======================================================================= 9.67  answers, alignment, Phase 4 prepared
def s967():
    a = J("E17_ALIGN_AT_BIRTH.json")
    sd = [str(s) for s in DEVSEEDS]
    cos = lambda m, l: [a[s][m][l]["cos"] for s in sd]
    chk("9.67-a embeddings: sign as run ; exact ; through time", ["0.13", "0.28", "0.32", "0.20", "0.33", "0.36", "0.51", "0.77", "0.53"],
        cos("sign", "book") + cos("exact", "book") + cos("sign_time", "book"))
    chk("9.67-b first layer: sign ; exact ; through time", ["0.24", "0.16", "0.32", "0.32", "0.24", "0.41", "0.47", "0.60", "0.45"],
        cos("sign", "l1") + cos("exact", "l1") + cos("sign_time", "l1"))
    chk("9.67-c first bank projection: sign ; trace ; exact ; through time",
        ["0.09", "-0.06", "0.05", "0.54", "0.61", "0.34", "0.08", "-0.01", "0.26", "0.81", "0.78", "0.55"],
        cos("sign", "lp") + cos("sign_tup", "lp") + cos("exact", "lp") + cos("sign_time", "lp"))
    chk_range("9.67-d first bank write (lc, lw): sign", "0.55", "0.67", cos("sign", "lc") + cos("sign", "lw"))
    chk_range("9.67-d' first bank write: exact", "0.59", "0.88", cos("exact", "lc") + cos("exact", "lw"))
    chk_range("9.67-d'' first bank write: through time", "0.71", "0.78", cos("sign_time", "lc") + cos("sign_time", "lw"))
    chk_range("9.67-e second bank projection: sign", "0.52", "0.57", cos("sign", "lp2"))
    chk_range("9.67-e' second bank projection: trace", "0.78", "0.80", cos("sign_tup", "lp2"))
    chk_range("9.67-e'' second bank projection: exact", "0.57", "0.66", cos("exact", "lp2"))
    chk_range("9.67-e''' second bank projection: through time", "0.78", "0.80", cos("sign_time", "lp2"))
    up = [x for l in ("lc2", "lw2", "lm", "lm2", "l2") for x in cos("sign", l)]
    chk_range("9.67-f second bank write, readout: sign (lc2, lw2, lm, lm2, l2)", "0.65", "0.81", up,
              note="output layer l3 has cosine 1.00 and is not in this range")
    allt = [x for l in ("book", "l1", "lp", "lc", "lw", "lp2", "lc2", "lw2", "lm", "lm2", "l2") for x in cos("sign_time", l)]
    chk_range("9.67-g every layer under through-time credit", "0.45", "0.81", allt, note="excluding output layer l3 (1.00)")
    c = J("E17P4_CHECK.json")
    chk("9.67-h Phase 4 option vs plain rule, max relative error ('2e-7')", ["2"], [max(c["downstream_plain_vs_option_max_rel_err"].values()) * 1e7], note="x 1e7")
    chk_fact("9.67-h' option equals direct computation exactly", "0", str(c["option_vs_direct_max_abs"]), c["option_vs_direct_max_abs"] == 0.0)
    chk("9.67-h'' ms per update: new arm; plain rule; backprop", ["48.7", "54.7", "49.4"], [c["ms_per_update_all"]["T"], c["ms_per_update_plain"]["T"], c["ms_per_update_plain"]["D"]])
    tw = "D_bank2_nbd"
    chk("9.67-i comparator D_bank2_nbd mod-97 ; mod-73 after block 6", ["92.3", "93.7", "93.3", "92.6", "91.6", "91.9"],
        [fm(last(e17(tw, s)), F1) for s in DEVSEEDS] + [fm(last(e17(tw, s)), F3) for s in DEVSEEDS])
    dr = {k: J(f"archive_e17p4_dry/e17_life_{k}_dry4_73129.json") for k in ("T_bank2_nbd_tup_raw65536", "T_bank2_nbd_tall_raw65536", "M_bank2_nbd_raw65536")}
    zz = {k: [x["settings"] for x in v["stages"] if "settings" in x][-1]["zero_grad_first_update"]["book_rows_in_batch_without_gradient"] for k, v in dr.items()}
    ns = {k: len(v["stages"]) for k, v in dr.items()}
    chk_fact("9.67-j dry runs: seven stages; rows without gradient one-layer trace / new arm", "7; 36 of 47 / 2",
             f"{sorted(set(ns.values()))}; {zz['T_bank2_nbd_tup_raw65536']} / {zz['T_bank2_nbd_tall_raw65536']} (backprop {zz['M_bank2_nbd_raw65536']})",
             set(ns.values()) == {7} and zz["T_bank2_nbd_tup_raw65536"] == 36 and zz["T_bank2_nbd_tall_raw65536"] == 2)
    fz = {s: (settings(T_drift(s)).get("freeze"), settings(T_drift(s)).get("frozen")) for s in S18}
    n_ok = sum(v == ("none", []) for v in fz.values())
    chk_fact("9.67-l window learner: 'freeze: none, frozen: [] in every receipt' (18 drift learners)", "18 of 18",
             f"{n_ok} of 18 carry both fields as stated; {sum(v[0] is None for v in fz.values())} have no freeze field, {sum(v[1] is None for v in fz.values())} no frozen field",
             n_ok == 18, note="absent fields are older receipts written before the option existed; none records freezing")
    nr("9.67-k forecast 2 h 20 min", "2 h 20 min", "a forecast, not a measurement")


# ======================================================================= 9.70  E17 Phase 4
def s970():
    NA, OL, BP, TW = "T_bank2_nbd_tall_raw65536", "T_bank2_nbd_tup_raw65536", "M_bank2_nbd_raw65536", "D_bank2_nbd"
    table = {
        (NA, 73129): ["92.9", "90.2", "82.9", "80.6", "86.0", "87.9", "90.8", "0.95", "0.98"],
        (NA, 93683): ["92.1", "93.0", "84.9", "82.0", "82.2", "86.1", "86.9", "0.92", "0.95"],
        (NA, 116089): ["90.7", "92.4", "75.5", "81.6", "83.9", "86.9", "89.4", "0.93", "0.97"],
        (OL, 73129): ["65.0", "65.5", "35.6", "31.8", "41.9", "44.2", "49.9", "0.48", "0.54"],
        (OL, 93683): ["75.3", "75.7", "40.0", "29.1", "42.1", "42.1", "47.2", "0.45", "0.52"],
        (OL, 116089): ["74.9", "74.2", "39.2", "37.6", "41.4", "42.6", "46.9", "0.46", "0.51"],
        (BP, 73129): ["93.6", "93.2", "87.6", "89.4", "92.2", "90.9", "93.5", "0.98", "1.01"],
        (BP, 93683): ["94.5", "93.2", "90.8", "86.1", "92.8", "92.3", "93.2", "0.99", "1.02"],
        (BP, 116089): ["93.8", "94.0", "90.2", "92.4", "93.4", "94.2", "93.2", "1.01", "1.01"]}
    for (arm, s), c in table.items():
        st = stages(e17(arm, s))
        t, d = last(e17(arm, s)), last(e17(TW, s))
        vals = [fm(st[b]["scores"], F1) for b in range(1, 7)] + [fm(t, F3), fm(t, F1) / fm(d, F1), fm(t, F3) / fm(d, F3)]
        chk(f"9.70-a table row {arm.split('_raw')[0]} {s} (blocks 1-6, mod-73, ratios)", c, vals)
    chk_range("9.70-b twin block 1", "92.2", "93.6", [blockF1(TW, s, 1) for s in DEVSEEDS])
    chk_range("9.70-b' twin block 6", "92.3", "93.7", [blockF1(TW, s, 6) for s in DEVSEEDS])
    chk_range("9.70-b'' twin mod-73", "91.6", "92.6", [fm(last(e17(TW, s)), F3) for s in DEVSEEDS])
    U = "T_bank2_nbd_raw65536"
    chk_range("9.70-c sign rule as run (Phase 2) block 1", "1.5", "2.8", [blockF1(U, s, 1) for s in DEVSEEDS])
    chk_range("9.70-c' ... block 6", "0.7", "1.2", [blockF1(U, s, 6) for s in DEVSEEDS])
    chk_range("9.70-c'' ... mod-73 at 6", "1.1", "2.2", [fm(last(e17(U, s)), F3) for s in DEVSEEDS])
    chk_range("9.70-d P17.4.5 held pairs new arm", "12.8", "14.7", [comp(last(e17(NA, s))) for s in DEVSEEDS])
    chk_range("9.70-d' ... one-layer", "7.7", "9.0", [comp(last(e17(OL, s))) for s in DEVSEEDS])
    chk_range("9.70-d'' ... backprop", "14.0", "18.5", [comp(last(e17(BP, s))) for s in DEVSEEDS])
    chk_range("9.70-d''' hard pairs (mean of +- and -+), three arms", "0.8", "8.6", [hard(last(e17(a, s))) for a in (NA, OL, BP) for s in DEVSEEDS])
    chk_range("9.70-e new arm block 1 ('91-93')", "91", "93", [blockF1(NA, s, 1) for s in DEVSEEDS])
    chk_fact("9.70-e' new arm ends above Phase 3 frozen trace (80.1/84.9/84.9) in every seed", "every seed",
             ", ".join(rnd(blockF1(NA, s, 6) - blockF1("T_bank2_nbd_frz1_tup_raw65536", s, 6), 1) for s in DEVSEEDS),
             all(blockF1(NA, s, 6) > blockF1("T_bank2_nbd_frz1_tup_raw65536", s, 6) for s in DEVSEEDS))
    chk_range("9.70-f backprop above new arm, mod-97", "3", "7", [blockF1(BP, s, 6) - blockF1(NA, s, 6) for s in DEVSEEDS])
    chk_range("9.70-f' backprop above new arm, mod-73", "3", "6", [fm(last(e17(BP, s)), F3) - fm(last(e17(NA, s)), F3) for s in DEVSEEDS])
    chk_range("9.70-f'' add/sub mod 97: new arm", "77", "86", [op(last(e17(NA, s)), n) for s in DEVSEEDS for n in ("add97", "sub97")])
    chk_range("9.70-f''' add/sub mod 97: backprop", "88", "93", [op(last(e17(BP, s)), n) for s in DEVSEEDS for n in ("add97", "sub97")])
    chk_range("9.70-g max/min mod 97: new arm", "91", "94", [op(last(e17(NA, s)), n) for s in DEVSEEDS for n in ("max97", "min97")])
    chk_range("9.70-g' max/min mod 97: backprop", "92", "97", [op(last(e17(BP, s)), n) for s in DEVSEEDS for n in ("max97", "min97")])
    rr = [fm(last(e17(BP, s)), f) / fm(last(e17(TW, s)), f) for s in DEVSEEDS for f in (F1, F3)]
    chk_range("9.70-g'' pipeline cost to backprop (ratios to twin)", "0.98", "1.02", rr)
    chk_range("9.70-h one-layer trace block 1", "65", "75", [blockF1(OL, s, 1) for s in DEVSEEDS])
    chk_range("9.70-h' one-layer trace end mod-97", "42", "44", [blockF1(OL, s, 6) for s in DEVSEEDS])
    chk_range("9.70-h'' one-layer trace end mod-73", "47", "50", [fm(last(e17(OL, s)), F3) for s in DEVSEEDS])
    chk_range("9.70-h''' one-layer loss when compositions enter (block 2 - min(blocks 3, 4))", "30", "46",
              [blockF1(OL, s, 2) - min(blockF1(OL, s, 3), blockF1(OL, s, 4)) for s in DEVSEEDS],
              note="block 2 - block 3 alone gives 29.9-35.7")
    z = [settings(e17(a, s))["zero_grad_first_update"]["book_rows_in_batch_without_gradient"] for a in (OL, NA) for s in DEVSEEDS]
    chk_fact("9.70-i rows without gradient: one-layer 36 of 47; new arm 2", "36 / 2", f"{z[:3]} / {z[3:]}", z[:3] == [36] * 3 and z[3:] == [2] * 3)
    chk("9.70-j development ops after block 6: new arm ; backprop ; twin",
        ["10.7", "22.2", "43.2", "48.6", "41.4", "53.6", "49.1", "37.9", "51.5"],
        [fm(last(e17(a, s)), DEV) for a in (NA, BP, TW) for s in DEVSEEDS])
    for lab, arm, lo, hi in (("new arm", NA, "6,672", "6,787"), ("one-layer", OL, "7,034", "7,068"), ("backprop", BP, "6,615", "6,732")):
        chk_range(f"9.70-k CPU-s per life, {lab}", lo, hi, [stages(e17(arm, s))[-1]["cpu"] for s in DEVSEEDS])
    bh = {s: {settings(e17(a, s)).get("birth_sha256") for a in (NA, OL, BP)} for s in DEVSEEDS}
    ok = all(len(v) == 1 and None not in v for v in bh.values()) and all(len(stages(e17(a, s))) == 7 for a in (NA, OL, BP) for s in DEVSEEDS)
    chk_fact("9.70-l nine lives loaded the same three births, seven stages", "same birth per seed; 7 stages",
             "birth hashes per seed: " + ", ".join(f"{s}:{len(v)} distinct" for s, v in bh.items()), ok)
    se = {a: settings(e17(a, 73129)) for a in (NA, OL, BP, TW)}
    chk_fact("9.70-m settings: new arm sign/all/none frozen; one-layer sign/proj; backprop rule D wd 0.25 rshare 0.5; twin wd 0.3",
             "as described", "; ".join(f"{a.split('_raw')[0]}: rule={v.get('rule')} fb={v.get('feedback')} traceup={v.get('traceup')} freeze={v.get('freeze')} wd={v.get('wd')} rshare={v.get('rshare')} projfb={v.get('projfb')}" for a, v in se.items()),
             se[NA]["traceup"] == "all" and se[NA]["feedback"] == "sign" and se[NA]["freeze"] == "none" and se[OL]["traceup"] == "proj"
             and se[BP]["rule"] == "D" and se[BP]["wd"] == 0.25 and se[TW]["wd"] == 0.3)
    nr("9.70-n wall 2 h 04 min; forecasts", "2 h 04 min", "wall time and forecasts are in logs, not JSON receipts")


# ======================================================================= 9.75  Codex's source audit, errata
def s975():
    st = {}
    for s in S18:
        a, b = last(T_drift(s)), last(D_drift(s))
        cor = "keyed by split" in settings(T_drift(s)).get("panels", "")
        st[s] = (cor, min(ratios(a, b)) >= .8, hard_ratio(a, b) >= .8, whole_ratio(a, b) >= .8)
    c = [v for v in st.values() if v[0]]
    o = [v for v in st.values() if not v[0]]
    chk_fact("9.75-a corrected panels: family / hard pairs / whole answer", "13 of 15; 2 of 15; 9 of 15",
             f"{sum(v[1] for v in c)} of {len(c)}; {sum(v[2] for v in c)} of {len(c)}; {sum(v[3] for v in c)} of {len(c)}",
             (len(c), sum(v[1] for v in c), sum(v[2] for v in c), sum(v[3] for v in c)) == (15, 13, 2, 9))
    chk_fact("9.75-a' original panels (73129, 93683, 116089): family / hard / whole", "3 of 3; 2 of 3; 3 of 3",
             f"{sum(v[1] for v in o)} of {len(o)}; {sum(v[2] for v in o)} of {len(o)}; {sum(v[3] for v in o)} of {len(o)}",
             (len(o), sum(v[1] for v in o), sum(v[2] for v in o), sum(v[3] for v in o)) == (3, 3, 2, 3))
    ex = {i: stages("e16_life_T_raw65536_drift_online_cp_73129.json")[i]["exposure_rows_tokens"] for i in (5, 6)}
    chk_fact("9.75-b fresh compositions arrive after checkpoint 4 and stop before the last block", "fresh F2 until ~3/4 of life",
             f"block 5 fresh/F2 rows {ex[5].get('fresh/F2', [0])[0]:,}; block 6 fresh/F2 rows {ex[6].get('fresh/F2', [0])[0]:,}; replay/F2 rows in block 6 {ex[6].get('replay/F2', [0])[0]:,} (73129 _cp receipt)",
             ex[5].get("fresh/F2", [0])[0] > 0 and "fresh/F2" not in ex[6],
             note="only per-block totals are recorded; the exact 'three quarters' point and the equal weights at checkpoint 4 are not recomputable")
    chk("9.75-c admitted rows at share 0.25 / 0.5 / 0.75 (400009)", ["1,769,488", "1,179,680", "589,872"],
        [stages(M3("T", sh, 400009))[-1]["admitted"] for sh in (0.25, 0.5, 0.75)])
    chk("9.75-d E16-M2 600013 checkpoint-0 compositions: learner/reference (original panel) ; M/oracle (corrected)", ["25.2", "24.6"],
        [comp(stages("e16_life_T_raw65536_600013.json")[0]["scores"]), comp(stages("e16_life_M_raw65536_600013.json")[0]["scores"])])
    chk_fact("9.75-d' panel versions of the four E16-M2 600013 arms", "T, D original; M, oracle corrected",
             ", ".join(f"{n.split('life_')[1].split('_600013')[0]}={'corrected' if 'keyed by split' in settings(n).get('panels', '') else 'original'}"
                       for n in ("e16_life_T_raw65536_600013.json", "e16_life_D_600013.json", "e16_life_M_raw65536_600013.json", "e16_life_T_oracle_600013.json")),
             ["keyed by split" in settings(n).get("panels", "") for n in ("e16_life_T_raw65536_600013.json", "e16_life_D_600013.json", "e16_life_M_raw65536_600013.json", "e16_life_T_oracle_600013.json")] == [False, False, True, True])
    NA, TW = "T_bank2_nbd_tall_raw65536", "D_bank2_nbd"
    chk("9.75-e(5a) Phase 4 development ops after block 1: new arm ; reference", ["59.2", "47.0", "60.2", "62.8", "58.0", "67.1"],
        [fm(stages(e17(NA, s))[1]["scores"], DEV) for s in DEVSEEDS] + [fm(stages(e17(TW, s))[1]["scores"], DEV) for s in DEVSEEDS])
    chk("9.75-e'(5a) ... at the end (new arm)", ["10.7", "22.2", "43.2"], [fm(last(e17(NA, s)), DEV) for s in DEVSEEDS])
    chk("9.75-f(5b) new arm at 40 distractors: add ; sub (final answer)", ["18.2", "9.2", "28.2", "16.8", "25.8", "19.0"],
        [op(last(e17(NA, s)), "add97", "40", "final") for s in DEVSEEDS] + [op(last(e17(NA, s)), "sub97", "40", "final") for s in DEVSEEDS])
    chk_range("9.75-f'(5b) ... against 77-86 at 12", "77", "86", [op(last(e17(NA, s)), n) for s in DEVSEEDS for n in ("add97", "sub97")])
    nr("9.75-g(5c) one-layer arm relayed raw signs, every-layer arm signs scaled by the mean weight", "raw / scaled",
       "the Phase 4 receipts have no projfb field (added after them); the claim rests on source code, which this script does not read")
    m = J("AUDIT_RESPONSE_CHECKS.json")["projection_feedback_map"]
    cosv = [m[str(s)]["cosine_raw_vs_scaled"][l] for s in DEVSEEDS for l in ("lp.weight", "lp2.weight")]
    chk_range("9.75-g'(5c) cosine raw vs scaled projection gradients", "0.991", "0.999", cosv)
    chk("9.75-g''(5c) alignment with true gradient, raw ; scaled (lp)", ["0.55", "0.66", "0.37", "0.54", "0.67", "0.38"],
        [m[str(s)]["raw_sign_as_run"]["lp.weight"] for s in DEVSEEDS] + [m[str(s)]["scaled_sign"]["lp.weight"] for s in DEVSEEDS])
    ind = J("E17P4_INDEPENDENT_CHECK.json")
    mx = max(v["max"] for v in ind["per_seed"].values())
    chk("9.75-h(6) independent check, max error ('2e-15')", ["2"], [mx * 1e15], note="x 1e15")
    nr("9.75-h'(6) start-of-sequence row never receives a gradient", "never", "stated from the independent check's run; the JSON holds per-tensor errors only")
    cen = J("AUDIT_RESPONSE_CHECKS.json")["census_at_init"]
    sym = lambda k: sum(v for r, v in cen[k]["rows_with_gradient"].items() if r != "BOS/EOS")
    tot = sum(v for r, v in cen["B3"]["rows_present"].items() if r != "BOS/EOS")
    chk_fact("9.75-i(7) E18 census at init: B3 symbols with gradient; B0/B1/B2", "11 of 45; 45 / 45 / 45",
             f"{sym('B3')} of {tot}; {sym('B0')} / {sym('B1')} / {sym('B2')}", sym("B3") == 11 and tot == 45 and sym("B0") == sym("B1") == sym("B2") == 45)
    so = J("AUDIT_RESPONSE_CHECKS.json")["split_overlap"]
    rev = [v["share"] for s in so for k, v in so[s].items() if "quarter taught" in k]
    rev89 = [v["share"] for s in so for k, v in so[s].items() if "half taught" in k]
    sub = [v["share"] for s in so for k, v in so[s].items() if "sub-step" in k]
    chk_about("9.75-j(9) reversal in taught pool, mod-97 and mod-73 ('25%'; measured on seeds 73129, 400009, 1200007 only)", "25", [100 * x for x in rev])
    chk_range("9.75-j'(9) ... at birth (mod 89)", "49", "52", [100 * x for x in rev89])
    chk_range("9.75-j''(9) held mod-97 pairs supervised as a sub-step", "71", "73", [100 * x for x in sub])
    nr("9.75-k(9) strict re-score: mod-97 mean moves -1.2 to +1.7 (mean +0.2), ratios by <= 0.025; 1600033 0.805 -> 0.798",
       "-1.2..+1.7; 0.025; 0.798", "made from saved weights (.pt); the JSON part was removed from E16_AUDIT_TABLES.json; withdrawn by 9.79 item 3")
    chk("9.75-k'(9) 1600033 registered mod-97 ratio (receipt)", ["0.805"], [ratios(last(T_drift(1600033)), last(D_drift(1600033)))[0]])
    nr("9.75-l(10) receipt and no-store scores identical in all fifteen seeds (single ops, k = 12)", "identical",
       "the fifteen cloud receipts carry no no-store field; the comparison was made from weights")


# ======================================================================= 9.76  composition from memory
def s976():
    rows = [("window reference (D_drift)", D_drift, (200003, 300007, 400009),
             ["92.8", "91.5", "92.5", "88.8", "85.2", "88.8", "98.3", "96.2", "98.0"]),
            ("window learner (T drift)", T_drift, (200003, 300007, 400009),
             ["77.5", "84.5", "82.8", "67.8", "66.0", "44.7", "89.1", "88.2", "76.0"]),
            ("windowless reference (D_bank2_nbd)", lambda s: e17("D_bank2_nbd", s), DEVSEEDS,
             ["48.7", "58.5", "51.7", "0.8", "6.8", "2.9", "22.2", "23.1", "25.4"]),
            ("windowless backprop on pipeline (M_bank2_nbd)", lambda s: e17("M_bank2_nbd_raw65536", s), DEVSEEDS,
             ["36.3", "42.5", "40.2", "0.8", "3.8", "1.8", "19.0", "20.0", "26.8"]),
            ("windowless weight signs through time (T tall)", lambda s: e17("T_bank2_nbd_tall_raw65536", s), DEVSEEDS,
             ["16.5", "24.2", "25.5", "0.4", "2.4", "1.0", "19.9", "20.4", "20.7"])]
    for lab, f, seeds, c in rows:
        sc = [last(f(s)) for s in seeds]
        vals = [pooled(x, ["within"], "exact") for x in sc] + [pooled(x, ["+-", "-+"], "exact") for x in sc] + [pooled(x, ["><", "<>"], "exact") for x in sc]
        chk(f"9.76-a table row: {lab} (within ; +- -+ ; >< <>, whole answers)", c, vals)
    sc = [last(e17("D_bank2_nbd", s)) for s in DEVSEEDS]
    chk_range("9.76-b windowless reference, untaught plus-minus pairs ('1-7%')", "1", "7", [pooled(x, ["+-", "-+"], "exact") for x in sc])
    ce = [100 * stages(e17("D_bank2_nbd", s))[0]["shortcut_ceilings"][p][k] for s in DEVSEEDS for p in ("><", "<>") for k in ("ignore_o1_use_a", "ignore_o1_use_b")]
    chk_range("9.76-c 'ignoring one operator' shortcut on comparison pairs", "66", "69", ce,
              note="shortcut_ceilings ignore_o1_use_a / _b, both pairs, three seeds")


# ======================================================================= 9.79  Codex's complete audit, errata; E19
def s979():
    sc = [last(e17("D_bank2_nbd", s)) for s in DEVSEEDS]
    chk_range("9.79-a(1) windowless reference on 'within' (new instances of taught pairs)", "49", "59", [pooled(x, ["within"], "exact") for x in sc])
    chk_range("9.79-a'(1) ... on untaught pairs", "1", "7", [pooled(x, ["+-", "-+"], "exact") for x in sc])
    sc_ = lambda s, h, k: 100 * last(T_drift(s))["silenced_compose"][h][k] / last(T_drift(s))["silenced_compose"][h]["n"]
    silpm = [sc_(s, h, "exact") for s in S18 for h in H[:2]]
    silx = [sc_(s, h, "exact") for s in S18 for h in H]
    silf = [sc_(s, h, "final") for s in S18 for h in H]
    chk_fact("9.79-b(2) window learner's held compositions with the state zeroed fall to zero", "zero",
             f"18 drift learners, per pair: whole answer on +-,-+ max {rnd(max(silpm), 1)}%, on all four pairs max {rnd(max(silx), 1)}%; final answer max {rnd(max(silf), 1)}%",
             max(silx) < 0.5, note="receipt field silenced_compose; the review does not say which metric or which learners")
    e = J("E16_EXPOSURE_STRATA.json")
    g = lambda o, k: e[f"mod 97 {o} | {k}"]
    chk("9.79-c(3) clean items: addition / max / min / subtraction", ["361", "125", "136", "1,656"],
        [g("add", "none (clean)")["n"], g("max", "none (clean)")["n"], g("min", "none (clean)")["n"], g("sub", "none (clean)")["n"]])
    chk("9.79-c'(3) registered items per operation (15 seeds x 400)", ["6,000"], [g("add", "ALL (registered panel)")["n"]])
    vals = []
    for o in ("add", "sub", "min", "max"):
        c, a = g(o, "none (clean)"), g(o, "ALL (registered panel)")
        vals += [100 * c["T"] / c["n"], 100 * a["T"] / a["n"]]
    chk("9.79-d(3) learner clean ; all: add, sub, min, max", ["75.6", "74.2", "73.7", "73.1", "94.9", "95.9", "89.6", "94.8"], vals)
    chk("9.79-d'(3) reference on clean items: add, sub, min, max", ["90.0", "90.5", "97.8", "97.6"],
        [100 * g(o, "none (clean)")["D"] / g(o, "none (clean)")["n"] for o in ("add", "sub", "min", "max")])
    chk("9.79-d''(3) max: learner clean below all by", ["5"], [100 * g("max", "ALL (registered panel)")["T"] / 6000 - 100 * g("max", "none (clean)")["T"] / 125])
    sup = [100 - 100 * g(o, "none (clean)")["n"] / g(o, "ALL (registered panel)")["n"] for o in ("add", "sub", "max", "min")]
    chk_range("9.79-e(3) registered mod-97 panel: share supervised somewhere in another form ('94-98%')", "94", "98", sup,
              note="add 94.0, sub 72.4, max 97.9, min 97.7; the 94-98 range holds for add, max, min only")
    chk_fact("9.79-f(4) corrected-panel family count: registered; with 1600033 at 0.798", "13 of 15; 12 of 15",
             f"{sum(min(ratios(last(T_drift(s)), last(D_drift(s)))) >= .8 for s in S18[3:])} of 15; strict value not recomputable",
             sum(min(ratios(last(T_drift(s)), last(D_drift(s)))) >= .8 for s in S18[3:]) == 13,
             note="12 of 15 follows arithmetically if 1600033 falls below 0.80; the 0.798 itself is from weights")
    nr("9.79-g E20-A v1 panel: first operand in view 6.6%, hidden 88.7%", "6.6% / 88.7%", "measured on the withdrawn v1 panel; recorded only in AUDIT_RESPONSE_2 (markdown), no JSON receipt")
    hm = lambda a, s: (lambda c: (op(c, "add89") + op(c, "sub89")) / 2)(J(f"e19_dev_kbank2_{a}_{s}.json")["curve"][-1]["sealed"])
    chk("9.79-h E19 linear-write birth by backprop, hard mean", ["69.0", "70.5", "69.3"], [hm("bptt", s) for s in DEVSEEDS])
    ref = [(lambda c: (op(c, "add89") + op(c, "sub89")) / 2)(J(f"e17_dev_bank2_nbd_{s}.json")["curve"][-1]["sealed"]) for s in DEVSEEDS]
    chk_range("9.79-h' the gated network ('91-95'): reference births e17_dev_bank2_nbd", "91", "95", ref)
    chk("9.79-i P19.3 forward-only rule with weight signs; ratio to bptt", ["43.2", "45.8", "54.5", "0.63", "0.65", "0.79"],
        [hm("r1", s) for s in DEVSEEDS] + [hm("r1", s) / hm("bptt", s) for s in DEVSEEDS])
    chk("9.79-j P19.4 r1 minus r2 (no deep traces)", ["8.0", "14.5", "23.8"], [hm("r1", s) - hm("r2", s) for s in DEVSEEDS])
    d = {a: J(f"e19_dev_kbank2_{a}_73129.json") for a in ("bptt", "r1", "r2")}
    chk_fact("9.79-j' settings: r1 and r2 same decay; bptt at a different decay", "r1 = r2; bptt differs",
             " / ".join(f"{a} decay {v['decay']}" for a, v in d.items()), d["r1"]["decay"] == d["r2"]["decay"] != d["bptt"]["decay"])
    ms = {a: [J(f"e19_dev_kbank2_{a}_{s}.json")["curve"][-1]["cpu_ms_per_update"] for s in DEVSEEDS] for a in ("r1", "bptt")}
    chk("9.79-k ms per update: rule ; backprop (mean of three)", ["220", "103"], [sum(ms["r1"]) / 3, sum(ms["bptt"]) / 3])


# ======================================================================= 9.71  E18 births
def s971():
    arms = {"ref": "e17_dev_bank2_nbd_{}.json", "B0": "e17_dev_bank2_nbd_bT_tall_{}.json", "B1": "e17_dev_bank2_nbd_lh_{}.json",
            "B2": "e17_dev_bank2_nbd_lh_bT_tall_{}.json", "B3": "e17_dev_bank2_nbd_lh_bT_{}.json"}
    C = {a: {s: J(p.format(s))["curve"] for s in DEVSEEDS} for a, p in arms.items()}
    hm = lambda c, k="12": (op(c, "add89", k) + op(c, "sub89", k)) / 2
    mm = lambda c: (op(c, "max89") + op(c, "min89")) / 2
    fin = lambda a: [C[a][s][-1]["sealed"] for s in DEVSEEDS]
    claimed = {"ref": (["91.2", "94.5", "90.3"], None, ["96.0", "95.2", "95.5"]),
               "B0": (["87.5", "90.7", "84.7"], ["84.5", "88.0", "74.7"], ["98.8", "97.8", "95.7"]),
               "B1": (["80.7", "82.2", "77.2"], ["79.2", "73.3", "73.5"], ["93.8", "95.7", "94.5"]),
               "B2": (["74.8", "69.7", "66.3"], ["71.2", "64.3", "62.2"], ["96.0", "96.7", "93.8"]),
               "B3": (["1.2", "0.8", "0.8"], None, ["0.8", "2.0", "2.8"])}
    for a, (h, k20, m) in claimed.items():
        f = fin(a)
        c = h + (k20 or []) + m
        v = [hm(x) for x in f] + ([hm(x, "20") for x in f] if k20 else []) + [mm(x) for x in f]
        chk(f"9.71-a table row {a} (hard mean{'; at 20' if k20 else ''}; max/min)", c, v)
    steps = {C[a][s][-1]["step"] for a in arms for s in DEVSEEDS if a != "ref"}
    meta = {a: {k: v for k, v in J(arms[a].format(73129)).items() if k in ("heads", "birthrule", "traceup", "decay", "feedback")} for a in arms if a != "ref"}
    chk_fact("9.71-b final step and package settings", "24,576; B0 top/T/all/0.25, B1 local/D/none/0.30, B2 local/T/all/0.25, B3 local/T/none/0.25",
             f"steps {sorted(steps)}; " + "; ".join(f"{a} {v.get('heads')}/{v.get('birthrule')}/{v.get('traceup')}/{v.get('decay')}" for a, v in meta.items()),
             steps == {24576} and (meta["B0"]["heads"], meta["B0"]["birthrule"], meta["B0"]["traceup"], meta["B0"]["decay"]) == ("top", "T", "all", 0.25)
             and (meta["B1"]["heads"], meta["B1"]["birthrule"], meta["B1"]["decay"]) == ("local", "D", 0.3)
             and (meta["B2"]["heads"], meta["B2"]["birthrule"], meta["B2"]["traceup"], meta["B2"]["decay"]) == ("local", "T", "all", 0.25)
             and (meta["B3"]["heads"], meta["B3"]["birthrule"], meta["B3"]["traceup"], meta["B3"]["decay"]) == ("local", "T", "none", 0.25))
    chk("9.71-c P18.1 first bank's own head, add89", ["27.3", "29.7", "28.3"], [op(C["B1"][s][-1]["head1"], "add89") for s in DEVSEEDS])
    r = [hm(a) / hm(b) for a, b in zip(fin("B0"), fin("ref"))]
    chk("9.71-d P18.3 B0 / reference", ["0.96", "0.96", "0.94"], r)
    chk("9.71-e P18.4 B0 - B1", ["6.8", "8.5", "7.5"], [hm(a) - hm(b) for a, b in zip(fin("B0"), fin("B1"))])
    chk("9.71-f P18.5 B2 - B1", ["-5.8", "-12.5", "-10.8"], [hm(a) - hm(b) for a, b in zip(fin("B2"), fin("B1"))])
    chk_range("9.71-g P18.6 B3 max/min below B2's by", "91", "95", [mm(a) - mm(b) for a, b in zip(fin("B2"), fin("B3"))])
    chk("9.71-h B0 - B2", ["12.7", "21.0", "18.4"], [hm(a) - hm(b) for a, b in zip(fin("B0"), fin("B2"))])
    cur = [x for s in DEVSEEDS for x in (hm(C["B0"][s][-2]["sealed"]), hm(C["B0"][s][-1]["sealed"]))]
    chk("9.71-i B0 still rising at the last checkpoint (previous -> last)", ["84", "88", "88", "91", "78", "85"], cur)
    h1 = [hm(C["B1"][s][-1]["head1"]) for s in DEVSEEDS]
    chk_about("9.71-j first bank's own head, hard mean ('about 20')", "20", h1)
    chk_about("9.71-j' upper bank adds ('about 60 points'): B1 hard mean minus head1 hard mean", "60", [hm(a) - b for a, b in zip(fin("B1"), h1)])
    chk_range("9.71-j'' first bank alone answers additions", "27", "30", [op(C["B1"][s][-1]["head1"], "add89") for s in DEVSEEDS])
    chk_about("9.71-j''' package (B1) under the end-to-end reference ('about 10 points')", "10", [hm(b) - hm(a) for a, b in zip(fin("B1"), fin("ref"))])
    mv = J("E18_EMBEDDING_MOVEMENT.json")["B3"]
    chk_fact("9.71-k B3 operator, letter, task-code rows unmoved; digit rows moved ~0.7 of a row norm", "0.00; 0.7",
             f"operator {mv['operator']}, letter {mv['distractor letter']}, task code {mv['task code']}; digit {mv['digit']}",
             all(x == 0 for k in ("operator", "distractor letter", "task code") for x in mv[k]) and all(same("0.7", x) for x in mv["digit"]))
    nr("9.71-l extra head of 657,153 parameters", "657,153", "the E18 birth receipts carry no parameter count")
    cpu = [C[a][s][-1]["cpu"] for a in ("B0", "B1", "B2", "B3") for s in DEVSEEDS]
    chk("9.71-m CPU-s for twelve births", ["65,118"], [sum(cpu)])
    nr("9.71-n 4 h 39 min wall; 5-6% over forecast", "4 h 39 min", "wall time is in logs, not JSON receipts")


# ======================================================================= 9.81  twelve repeats on corrected panels
def s981():
    cpT = lambda s: f"e16_life_T_raw65536_drift_online_cp_{s}.json"
    cpD = lambda s: f"e16_life_D_drift_cp_{s}.json"
    names = [cpT(s) for s in DEVSEEDS] + [cpD(s) for s in DEVSEEDS] + [f"e16_life_{a}_cp_{s}.json" for a in ("T_raw65536", "M_raw65536", "D") for s in (400009, 600013)]
    ok7 = all(len(stages(n)) == 7 and "complete" in J(n) and J(n)["complete"].get("weights_sha256") for n in names)
    chk_fact("9.81-a twelve receipts: seven stages and a completion record with weight hash", "12 of 12",
             f"{sum(len(stages(n)) == 7 and 'complete' in J(n) for n in names)} of {len(names)}", ok7 and len(names) == 12,
             note="that each weight hash matches its .pt file is not recomputed (no .pt opened)")
    ph = all(settings(cpT(s)).get("panel_sha256") == settings(cpD(s)).get("panel_sha256") for s in DEVSEEDS) and \
        all(len({settings(f"e16_life_{a}_cp_{s}.json").get("panel_sha256") for a in ("T_raw65536", "M_raw65536", "D")}) == 1 for s in (400009, 600013))
    chk_fact("9.81-b every comparison between receipts with the same panel hash", "same hash", "same hash" if ph else "hash differs", ph)
    c0 = {s: [comp(stages(f"e16_life_{a}_cp_{s}.json")[0]["scores"]) for a in ("T_raw65536", "M_raw65536", "D")] for s in (400009, 600013)}
    chk("9.81-c staged arms' checkpoint-0 composition score (400009; 600013)", ["24.17", "24.58"], [c0[400009][0], c0[600013][0]])
    chk_fact("9.81-c' ... identical across the three arms of each seed", "identical", str({s: [rnd(x, 2) for x in v] for s, v in c0.items()}), all(len(set(v)) == 1 for v in c0.values()))
    claimed = {73129: ["0.97", "0.86", "0.99", "0.81", "0.84"], 93683: ["0.93", "0.87", "0.96", "0.82", "0.84"], 116089: ["0.84", "0.86", "0.97", "0.84", "0.83"]}
    for s, c in claimed.items():
        a, b = last(cpT(s)), last(cpD(s))
        chk(f"9.81-d drift repeat {s}: ratios mod-97, held, mod-73; plus-minus; whole", c, ratios(a, b) + [hard_ratio(a, b), whole_ratio(a, b)])
    st = {s: [last(f"e16_life_{a}_cp_{s}.json") for a in ("T_raw65536", "M_raw65536", "D")] for s in (400009, 600013)}
    chk("9.81-e staged held-pair ratio: learner (400009, 600013); backprop on pipeline", ["0.83", "0.64", "0.98", "0.94"],
        [comp(st[s][0]) / comp(st[s][2]) for s in (400009, 600013)] + [comp(st[s][1]) / comp(st[s][2]) for s in (400009, 600013)])
    chk("9.81-e' plus-minus pairs learner / pipeline / reference (400009 ; 600013)", ["67.4", "86.8", "90.4", "46.6", "80.9", "89.2"],
        [hard(x) for s in (400009, 600013) for x in st[s]])
    allr = [x for s in DEVSEEDS for x in ratios(last(cpT(s)), last(cpD(s)))]
    chk_fact("9.81-f P-CP.1 'all nine ratios at or above 0.84'", ">= 0.84", f"min {rnd(min(allr), 3)}", min(allr) >= 0.84,
             note="0.837 rounds to 0.84; strictly below")
    nsl = []
    for n in [cpT(s) for s in DEVSEEDS] + [f"e16_life_T_raw65536_cp_{s}.json" for s in (400009, 600013)]:
        sc = last(n)
        nsl.append(abs(comp(sc) - sum(100 * sc["nostore_compose"][h]["final"] / sc["nostore_compose"][h]["n"] for h in H) / 4))
    chk_fact("9.81-g P-CP.4 held-pair mean with and without the store identical (five learner lives)", "identical to two decimals",
             f"max |difference| {max(nsl):.4f}", max(nsl) < 0.005)
    fam = hp = wh = 0
    for s in S18:
        tn, dn = (cpT(s), cpD(s)) if s in DEVSEEDS else (T_drift(s), D_drift(s))
        a, b = last(tn), last(dn)
        fam += min(ratios(a, b)) >= .8; hp += hard_ratio(a, b) >= .8; wh += whole_ratio(a, b) >= .8
    chk_fact("9.81-h eighteen seeds on one panel version: family / plus-minus / whole", "16 / 5 / 12 of 18", f"{fam} / {hp} / {wh} of 18", (fam, hp, wh) == (16, 5, 12))
    hp15 = sum(hard_ratio(last(T_drift(s)), last(D_drift(s))) >= .8 for s in S18[3:])
    chk_fact("9.81-i fifteen later seeds pass the hard-pair gate", "2 of 15", f"{hp15} of 15", hp15 == 2)
    same_ops = all(last(cpT(s))[n] == last(T_drift(s))[n] for s in DEVSEEDS for n in F1 + F3 + DEV)
    mv = [abs(comp(last(cpT(s))) - comp(last(T_drift(s)))) for s in DEVSEEDS]
    chk_fact("9.81-j repeats reproduce single-operation scores to the digit; held-pair means moved < 1 point", "identical; < 1",
             f"{'identical' if same_ops else 'differ'}; max move {rnd(max(mv), 2)}", same_ops and max(mv) < 1)
    for lab, arms_, lo, hi in (("learners", [cpT(s) for s in DEVSEEDS] + [f"e16_life_T_raw65536_cp_{s}.json" for s in (400009, 600013)], "6,275", "6,307"),
                               ("backprop on the pipeline", [f"e16_life_M_raw65536_cp_{s}.json" for s in (400009, 600013)], "5,718", "5,742"),
                               ("references", [cpD(s) for s in DEVSEEDS] + [f"e16_life_D_cp_{s}.json" for s in (400009, 600013)], "5,675", "5,807")):
        chk_range(f"9.81-k CPU-s, {lab}", lo, hi, [stages(n)[-1]["cpu"] for n in arms_])
    nr("9.81-l wall 2 h 17 min; lock released 16:44:44", "2 h 17 min", "wall clock and lock are in logs")


# ======================================================================= 9.77  E20-A stage 1
def s977():
    rows = {s: last(f"e16_life_D_cm6_{s}.json")["memory"] for s in DEVSEEDS}
    sil = {s: last(f"e16_life_D_cm6_{s}.json")["memory_silenced"] for s in DEVSEEDS}
    pm, cm = ("+-", "-+"), ("><", "<>")
    t = lambda m, g, ps, k: sum(m[g][p][k] for p in ps)
    chk("9.77-a plus-minus twins both right (of 300)", ["63", "80", "130"], [t(rows[s], "twins", pm, "both_correct") for s in DEVSEEDS])
    chk("9.77-a' ... per cent", ["21.0", "26.7", "43.3"], [100 * t(rows[s], "twins", pm, "both_correct") / 300 for s in DEVSEEDS])
    chk("9.77-b same twins, single items right", ["42.3", "47.7", "66.3"], [100 * t(rows[s], "twins", pm, "items_correct") / 600 for s in DEVSEEDS])
    chk("9.77-c in-view control (letters in front)", ["78.3", "70.0", "80.7"], [100 * t(rows[s], "lead", pm, "items_correct") / 600 for s in DEVSEEDS])
    chk("9.77-d comparison twins both right", ["63.7", "58.7", "56.0"], [100 * t(rows[s], "twins", cm, "both_correct") / 300 for s in DEVSEEDS])
    chk("9.77-e operand hidden, new instances ; taught triples", ["72.3", "72.3", "85.7", "82.7", "78.3", "84.3"],
        [100 * rows[s]["within_hidden"]["exact"] / rows[s]["within_hidden"]["n"] for s in DEVSEEDS] + [100 * rows[s]["taught_hidden"]["exact"] / rows[s]["taught_hidden"]["n"] for s in DEVSEEDS])
    chk("9.77-f twins with the state zeroed, single items", ["0.0", "0.0", "0.2"], [100 * t(sil[s], "twins", pm, "items_correct") / 600 for s in DEVSEEDS])
    chk("9.77-g P20.2 in-view minus single items", ["36.0", "22.3", "14.3"],
        [100 * (t(rows[s], "lead", pm, "items_correct") - t(rows[s], "twins", pm, "items_correct")) / 600 for s in DEVSEEDS])
    chk("9.77-h P20.3 one seed exactly on the mark (420 of 600)", ["420"], [t(rows[93683], "lead", pm, "items_correct")])
    n = [f"e16_life_D_cm6_{s}.json" for s in DEVSEEDS]
    man = J("BIRTH_MANIFEST.json")
    q = J("QUALIFY_E20_MEMORY_PANELS.json")["memory_panels"]["sha256"]
    ok = all(len(stages(x)) == 7 and "complete" in J(x) for x in n) and all(settings(x).get("birth_admission") == "manifest" and
             man.get(settings(x)["birth"], {}).get("sha256") == settings(x)["birth_sha256"] for x in n) and settings(n[0])["memory_panels"]["sha256"] == q \
        and all(settings(x).get("store") is False for x in n)
    chk_fact("9.77-i seven stages, completion record, birth admitted by manifest, 73129 memory-panel hash = qualification, no store",
             "all", "all" if ok else "not all", ok, note="weight-hash-to-file match not recomputed (no .pt opened)")
    chk_range("9.77-j registered letter-free plus-minus panel, whole answers", "69", "81", [hard(last(x), "exact") for x in n])
    cand = {nm: [hard(last(f.format(s)), "exact") for s in DEVSEEDS] for nm, f in
            (("staged D", "e16_life_D_{}.json"), ("D_drift", "e16_life_D_drift_{}.json"), ("D_drift_cp", "e16_life_D_drift_cp_{}.json"))}
    nr("9.77-j' 'the reference taught without letters ended at 85-89' (plus-minus, whole answers)", "85–89",
       "the review does not name the receipt; candidates: " + "; ".join(f"{k} {rnd(min(v), 1)}–{rnd(max(v), 1)}" for k, v in cand.items()))
    chk_range("9.77-k operand hidden, single items of untaught plus-minus pairs", "42", "66", [100 * t(rows[s], "twins", pm, "items_correct") / 600 for s in DEVSEEDS])


# ======================================================================= 9.68  E16-M4 incomplete
def s968():
    claimed = {1100009: ["62.2", "71.1", "89.7", "0.69", "0.79"], 1200007: ["45.2", "69.5", "89.9", "0.50", "0.77"],
               1300021: ["69.6", "76.4", "88.9", "0.78", "0.86"], 1400017: ["71.8", "76.5", "87.4", "0.82", "0.88"],
               1500007: ["71.3", "70.2", "90.2", "0.79", "0.78"], 1600033: ["64.6", "75.1", "81.6", "0.79", "0.92"]}
    k = lambda s: f"e16_life_T_raw65536_drift_online_rs75_{s}_killed.json"
    a5 = {s: stages(T_drift(s))[5]["scores"] for s in claimed}
    b5 = {s: stages(k(s))[5]["scores"] for s in claimed}
    r5 = {s: stages(D_drift(s))[5]["scores"] for s in claimed}
    for s, c in claimed.items():
        chk(f"9.68-a ck5 seed {s}: share 0.5, 0.75, reference, ratios", c,
            [hard(a5[s]), hard(b5[s]), hard(r5[s]), hard(a5[s]) / hard(r5[s]), hard(b5[s]) / hard(r5[s])])
    ends = {s: len(stages(k(s))) - 1 for s in claimed}
    for s in (300007, 400009, 600013):
        for t in ("88", "94"):
            ends[f"{s}_rs{t}"] = len(stages(f"e16_life_T_raw65536_drift_online_rs{t}_{s}_killed.json")) - 1
    nocomp = all("complete" not in J(k(s)) for s in claimed)
    chk_fact("9.68-b twelve receipts end at checkpoint 5, no completion record", "12 at ck5", f"{sum(v == 5 for v in ends.values())} of {len(ends)} at ck5; completion record absent: {nocomp}",
             all(v == 5 for v in ends.values()) and len(ends) == 12 and nocomp)
    d = [hard(b5[s]) - hard(a5[s]) for s in claimed]
    chk_fact("9.68-c higher at 0.75 in five of six; by >= 5 in four", "5; 4", f"{sum(x > 0 for x in d)}; {sum(x >= 5 for x in d)}", sum(x > 0 for x in d) == 5 and sum(x >= 5 for x in d) == 4)
    chk_fact("9.68-d at or above 0.80 of reference: share 0.75; share 0.5", "3 of 6; 1 of 6",
             f"{sum(hard(b5[s]) / hard(r5[s]) >= .8 for s in claimed)} of 6; {sum(hard(a5[s]) / hard(r5[s]) >= .8 for s in claimed)} of 6",
             sum(hard(b5[s]) / hard(r5[s]) >= .8 for s in claimed) == 3 and sum(hard(a5[s]) / hard(r5[s]) >= .8 for s in claimed) == 1)
    g75 = sum(min(ratios(b5[s], r5[s])) >= .8 for s in claimed)
    g50 = [s for s in claimed if min(ratios(a5[s], r5[s])) < .8]
    chk_fact("9.68-e family gate at ck5: all six at 0.75; at 0.5 two fail (1200007, 1600033)", "6; 1200007, 1600033", f"{g75}; {', '.join(map(str, g50))}", g75 == 6 and g50 == [1200007, 1600033])
    chk("9.68-e' 1200007 held-pair ratio at 0.5; 1600033 mod-97 ratio at 0.5", ["0.73", "0.79"], [ratios(a5[1200007], r5[1200007])[1], ratios(a5[1600033], r5[1600033])[0]])
    ab = lambda st, s: pr(st[4]["scores"], "+-") - pr(st[5]["scores"], "+-")
    chk("9.68-f +- fall ck4 -> ck5 in 1200007, 1300021 at share 0.5 ; at 0.75", ["29.2", "21.2", "7.3", "8.5"],
        [ab(stages(T_drift(s)), s) for s in (1200007, 1300021)] + [ab(stages(k(s)), s) for s in (1200007, 1300021)])
    cl = {400009: ["51.5", "89.7", "69.8", "87.8", "70.1", "82.4", "72.2", "82.1"],
          600013: ["46.2", "80.8", "56.9", "80.4", "69.2", "77.4", "73.9", "78.3"],
          300007: ["68.8", "89.9", "75.4", "89.8", "56.5", "87.5", "79.8", "88.0"]}
    for s, c in cl.items():
        v = []
        for sh in (0.5, 0.75, 0.875, 0.9375):
            n = M3("T", sh, s) if sh in (0.5, 0.75) else f"e16_life_T_raw65536_drift_online{TAG[sh]}_{s}_killed.json"
            sc = stages(n)[5]["scores"]
            v += [hard(sc), fm(sc, F3)]
        chk(f"9.68-g curve beyond 0.75, seed {s} (hard; mod-73 at 0.5/0.75/0.875/0.9375)", c, v)
    drop = []
    gate_ok = True
    for s in cl:
        f75 = fm(stages(M3("T", .75, s))[5]["scores"], F3)
        for t in ("88", "94"):
            sc = stages(f"e16_life_T_raw65536_drift_online_rs{t}_{s}_killed.json")[5]["scores"]
            drop.append(f75 - fm(sc, F3))
            gate_ok &= min(ratios(sc, stages(D_drift(s))[5]["scores"])) >= .8
    chk_range("9.68-h mod-73 lower at the two highest shares than at 0.75", "2", "6", drop, note="against share 0.5 the drop is 1.8-7.6")
    chk_fact("9.68-h' at ck5 no high-share life under the family gate", "none under", "none under" if gate_ok else "some under", gate_ok)


# ======================================================================= coordinator note, section 2
def scn():
    c = [s for s in S18[3:]]
    fam = sum(min(ratios(last(T_drift(s)), last(D_drift(s)))) >= .8 for s in c)
    hp = sum(hard_ratio(last(T_drift(s)), last(D_drift(s))) >= .8 for s in c)
    wh = sum(whole_ratio(last(T_drift(s)), last(D_drift(s))) >= .8 for s in c)
    fz = [str(settings(T_drift(s)).get("freeze")) for s in S18]
    chk_fact("CN2-1 window learner, nothing frozen: family / plus-minus / whole on 15 corrected-panel seeds", "13 / 2 / 9 of 15; nothing frozen",
             f"{fam} / {hp} / {wh} of 15; freeze fields {sorted(set(fz))}", (fam, hp, wh) == (13, 2, 9) and set(fz) <= {"none", "None"},
             note="'None' = field absent in older receipts; no receipt records any freezing")
    chk_fact("CN2-2 retains nothing without replay (share 0)", "~1% hard pairs; 0.13-0.35 mod-97 ratio",
             ", ".join(rnd(hard(last(M3("T", 0.0, s))), 1) for s in (400009, 600013, 300007)), all(hard(last(M3("T", 0.0, s))) < 2 for s in (400009, 600013, 300007)))
    gr = [hard(last(M3("T", .75, s))) - hard(last(M3("T", .25, s))) for s in (400009, 600013)]
    gb = [hard(last(M3("M", .75, s))) - hard(last(M3("M", .25, s))) for s in (400009, 600013)]
    chk_fact("CN2-3 more replay -> better hard pairs, rule more than backprop", "rule gain > backprop gain",
             f"rule +{rnd(gr[0], 1)}/+{rnd(gr[1], 1)}, backprop +{rnd(gb[0], 1)}/+{rnd(gb[1], 1)}", all(a > b > 0 for a, b in zip(gr, gb)))
    a = J("E17_ALIGN_AT_BIRTH.json")
    lp = {m: [a[str(s)][m]["lp"]["cos"] for s in DEVSEEDS] for m in ("sign", "exact", "sign_time")}
    chk_fact("CN2-4 first-bank projection: truncation in time, not weight signs (alignment at birth)", "exact feedback no better; through-time much better",
             f"sign {[rnd(x, 2) for x in lp['sign']]}, exact {[rnd(x, 2) for x in lp['exact']]}, through time {[rnd(x, 2) for x in lp['sign_time']]}",
             max(lp["exact"]) < 0.3 and min(lp["sign_time"]) > 0.5)
    NA, TW, BP = "T_bank2_nbd_tall_raw65536", "D_bank2_nbd", "M_bank2_nbd_raw65536"
    chk_range("CN2-5 Phase 4 new arm, mod-97 ratio to reference", "0.92", "0.95", [blockF1(NA, s, 6) / blockF1(TW, s, 6) for s in DEVSEEDS])
    chk_range("CN2-5' original skills fall to", "11", "43", [fm(last(e17(NA, s)), DEV) for s in DEVSEEDS])
    chk_range("CN2-6 backprop on the pipeline above the new arm, mod-97 ('3-7 points')", "3", "7", [blockF1(BP, s, 6) - blockF1(NA, s, 6) for s in DEVSEEDS])
    chk_range("CN2-6' backprop on the pipeline / reference", "0.98", "1.02", [fm(last(e17(BP, s)), f) / fm(last(e17(TW, s)), f) for s in DEVSEEDS for f in (F1, F3)])
    e = J("E19_CHECK.json")
    m64 = max(i["test1_max"] for i in e["inits"] if i["dtype"].endswith("float64"))
    m32 = max(i["test1_max"] for i in e["inits"] if i["dtype"].endswith("float32"))
    chk_fact("CN2-7 forward-only credit matches backprop to rounding (worker's E19_CHECK.json)", "to rounding",
             f"max relative error {m64:.1e} in float64 (3 inits), {m32:.1e} in float32 (3 inits); all test1/test2 pass: {e['test1_pass_all']}/{e['test2_pass_all']}",
             m64 < 1e-14 and m32 < 1e-6 and e["test1_pass_all"] and e["test2_pass_all"],
             note="the coordinator's own independent check is recorded only in E19_COORDINATOR_CHECK.md (no JSON)")
    hm = lambda f: (lambda c: (op(c, "add89") + op(c, "sub89")) / 2)(J(f)["curve"][-1]["sealed"])
    cost = [hm(f"e17_dev_bank2_nbd_{s}.json") - hm(f"e19_dev_kbank2_bptt_{s}.json") for s in DEVSEEDS]
    chk("CN2-7' linear write costs about 22 points of birth accuracy under backprop (mean of three)", ["22"], [sum(cost) / 3],
        note=f"per seed {', '.join(rnd(x, 1) for x in cost)}; reference births on the owner's machine, E19 births on a cloud container")
    so = J("AUDIT_RESPONSE_CHECKS.json")["split_overlap"]
    rev = [100 * v["share"] for s in so for k, v in so[s].items() if "quarter taught" in k]
    sub = [100 * v["share"] for s in so for k, v in so[s].items() if "sub-step" in k]
    chk_about("CN2-8 reversal overlap ('25%')", "25", rev)
    chk_range("CN2-8' sub-step exposure", "71", "73", sub)
    nr("CN2-8'' strict panels move ratios by at most 0.025; 'did not inflate scores'", "<= 0.025",
       "from the strict re-score on weights, withdrawn by review 9.79 item 3; not in any JSON")


def main():
    for f in (s959, s960, s961, s966, s962, s964, s967, s970, s975, s976, s979, s971, s981, s977, s968, scn):
        try:
            f()
        except Exception as ex:  # a missing receipt is reported, not hidden
            ROWS.append((f.__name__, "", f"ERROR {type(ex).__name__}: {ex}", "NOT_RECOMPUTED"))
    if "--summary" not in sys.argv:
        print("claim-id | claimed | recomputed | match")
        for r in ROWS:
            print(" | ".join(str(x) for x in r))
    from collections import Counter
    c = Counter(r[3] for r in ROWS)
    print(f"\nSUMMARY: {len(ROWS)} checks; MATCH {c['MATCH']}; WITHIN {c['WITHIN']}; DIFFER {c['DIFFER']}; NOT_RECOMPUTED {c['NOT_RECOMPUTED']}")
    print("DIFFER:", "; ".join(r[0].split(" ")[0] for r in ROWS if r[3] == "DIFFER"))
    print("WITHIN:", "; ".join(r[0].split(" ")[0] for r in ROWS if r[3] == "WITHIN"))


if __name__ == "__main__":
    main()
