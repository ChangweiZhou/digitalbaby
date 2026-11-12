"""Sizing sweep, stage 1: step 0's checks 1 to 10, parameterized (a copy of ../target_step0_check.py; brief OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md rev. 3).

    python3 sz_check.py parity      world seeds 1, 2, 3 at the default configuration; writes SZ_PARITY_CHECK.json and compares it, value by
                                    value, with ../TARGET_STEP0_CHECK.json (exit code nonzero on any difference or any failed check)

Every check takes its configuration from the bundle it is given (B["cfg"], the one configuration the bundle was built from); the oracle and
the solver are constructed with that same configuration and each computes for itself. Changes from step 0's checker: the configuration is
passed instead of read from the generator's constants; parse() takes the configuration; the disclosure ledger is computed in one sweep
(same values, linear in the stream); check 2's separation test reads sz_oracle.py and sz_solver.py and also requires that neither imports
any module of the sweep; check 8's "other seed" fixture is given explicitly. The rest is step 0's text.

Step 0 docstring:

A check passes only when its condition holds on the real streams AND its fixture fails it. No learner is trained; nothing is concluded
about any learner. Three separate programs answer probes: the oracle (target_oracle.py: hidden values, never the stream), the prefix-only
solver (target_solver.py: bytes and the change point, never the hidden values), and the memorisation and recency tables below (bytes only).
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_oracle as OR      # noqa: E402
import sz_solver as SV      # noqa: E402
import sz_stream as G       # noqa: E402

SEEDS = (1, 2, 3)


# ---------------------------------------------------------------- helpers
def oracle_for(B, world=None):
    w = world or B["world"]
    return OR.Oracle(B["cfg"], w["v"], w["u"], w["u_new"], B["meta"]["change_at"])


def off(B, stream, at):
    o = B["offsets"][stream]
    return o["starts"][at] if at < len(o["starts"]) else o["end"]


def change_byte(B, stream):
    return B["offsets"][stream]["starts"][B["meta"]["change_at"]]


def walk(B, stream, fn, shared=False, probes=None):
    """Feed the prefix-only solver the stream up to each probe, in order, and call fn(solver, probe, prefix_end)."""
    s = SV.PrefixSolver(B["cfg"], change_byte(B, stream), shared_context_values=shared); data = B["streams"][stream]; pos = 0
    for p in sorted(probes or B["probes"], key=lambda p: off(B, stream, p["at"])):
        e = off(B, stream, p["at"]); s.feed(data[pos:e]); pos = e
        fn(s, p, e)
    return s


def parse(cfg, data):
    """Every episode and line in a stream, with its byte offset (checker's own reader; shares nothing with the solver)."""
    out, pos = [], 0; K = cfg["cues_per_family"]
    for tok in data.split(b"\n")[:-1]:
        k = 0
        while k < len(tok) and tok[k] in G.FILL:
            k += 1
        t = tok[k:]; at = pos + k; pos += len(tok) + 1
        if not t:
            continue
        x = t[0] - 65
        if t[1:2] == b":":
            cb = t[5]; out.append(dict(type="line", x=x, f=(cb - 97) // K, c=(cb - 97) % K, vv=int(t[2:4]), uu=int(t[7:9]), off=at))
        else:
            q = t.index(b"="); cb = t[q - 1]
            out.append(dict(type="ep", x=x, f=(cb - 97) // K, c=(cb - 97) % K, o=int(t[q + 1:q + 3]), off=at, eq=at + q))
    return out


def panel_items(B):
    for (name, f), items in B["panels"].items():
        yield name, f, items


def panel_at(B, name):
    return B["meta"]["panel_probes"][name]


def result(ok, fixture_failed, **kw):
    return dict(passed=bool(ok and fixture_failed), condition_held=bool(ok), fixture_failed_as_required=bool(fixture_failed), **kw)


# ---------------------------------------------------------------- 1. identification from the prefix of the ordinary stream
def identification(B, stream="ordinary", shared=False):
    O = oracle_for(B); bad, scored, unscored, n_sealed = [], 0, [], 0
    eps = parse(B["cfg"], B["streams"][stream]); cb = change_byte(B, stream)

    def fn(s, p, end):
        nonlocal scored, n_sealed
        if p["kind"] == "first":
            f, x, c = p["f"], p["x"], p["c"]
            if p["set"] == "revise" and f == 0 and c in B["world"]["changed"]:     # declared evidence after the change, read from the prefix
                ctx = {e["x"] for e in eps if e["type"] == "ep" and cb <= e["off"] < end and e["f"] == 0 and e["c"] == c}
                n_ep = sum(1 for e in eps if e["type"] == "ep" and cb <= e["off"] < end and e["f"] == 0 and e["c"] == c)
                if len(ctx) < 3 or n_ep < 3:
                    unscored.append(dict(at=p["at"], f=f, x=x, c=c, contexts=len(ctx))); return
            a = s.query(f, x, c); scored += 1
            if a is None or a != O.answer(f, x, c, p["at"]):
                bad.append(dict(kind="first", at=p["at"], set=p["set"], f=f, x=x, c=c, solver=a, oracle=O.answer(f, x, c, p["at"])))
        else:
            for f in (0, 1):
                for tw in B["split"]["families"][f]["twins"]:
                    for (x, c) in tw:
                        a = s.query(f, x, c); n_sealed += 1
                        if a is None or a != O.answer(f, x, c, p["at"]):
                            bad.append(dict(kind="panel", name=p["name"], f=f, x=x, c=c, solver=a, oracle=O.answer(f, x, c, p["at"])))
    s = walk(B, stream, fn, shared)
    return dict(first_encounter_scored=scored, sealed_members_checked=n_sealed, not_determined_or_wrong=bad, revision_probes_not_scored=unscored,
                solver_inconsistent_rows=s.inconsistent)


def fixture_split_isolated(cfg, seed):
    """One context taught with one cue only, and that cue taught with no other context: the context's held pairs must be undetermined."""
    world = G.make_world(cfg, seed); split = G.make_split(cfg, world, seed); F = split["families"][0]
    ctl = set(split["control_contexts"])
    xs, cs = next(((x, c) for (x, c) in F["acquire"] if x not in ctl), None)
    cstar = next(c for (x, c) in F["taught"] if x == xs and c not in world["changed"] and c != cs)
    F["taught"] = [p for p in F["taught"] if p[0] != xs and p[1] != cstar] + [(xs, cstar)]
    # a reveal teaches: keep one held pair of that context, and none with its cue, so that nothing links it before its probe
    for s_ in ("acquire", "retain", "revise"):
        F[s_] = [p for p in F[s_] if p[1] != cstar and (p[0] != xs or p == (xs, cs))]
    B = G.build(cfg, seed, split_override=split)
    r = identification(B)
    held = {(x, c) for s_ in ("acquire", "retain", "revise") for (x, c) in F[s_] if x == xs}
    flagged = {(b["x"], b["c"]) for b in r["not_determined_or_wrong"] if b["kind"] == "first" and b["f"] == 0}
    return dict(context=chr(65 + xs), only_cue=chr(97 + cstar), held_pair=chr(65 + xs) + chr(97 + cs), its_held_pairs=len(held), reported_undetermined=len(held & flagged),
                all_reported=held <= flagged and bool(held), total_reported=len(r["not_determined_or_wrong"]))


def check1(B):
    r = identification(B)
    ok = not r["not_determined_or_wrong"] and not r["revision_probes_not_scored"] and r["solver_inconsistent_rows"] == 0
    fx = fixture_split_isolated(B["cfg"], B["seed"])
    info = {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()}
    info["details"] = r["not_determined_or_wrong"][:5] + r["revision_probes_not_scored"][:5]
    for stream in ("wider", "moved", "extra"):       # not required: the comparison streams hold at least as much
        rr = identification(B, stream); info[f"also_{stream}"] = len(rr["not_determined_or_wrong"]) + len(rr["revision_probes_not_scored"])
    return result(ok, fx["all_reported"], **info, fixture=dict(what="one context taught with a single cue that no other context shares", **fx))


# ---------------------------------------------------------------- 2. oracle
def oracle_scores(B, O):
    lost = set(); n_tw = n_first = 0
    for name, f, items in panel_items(B):
        at = panel_at(B, name)
        for k, d in enumerate(items):
            n_tw += 1; ok1 = O.answer(f, d["x1"], d["c"], at) == G.true_outcome(B["cfg"], B["world"], f, d["x1"], d["c"], int(at >= B["meta"]["change_at"]))
            ok2 = O.answer(f, d["x2"], d["c"], at) == G.true_outcome(B["cfg"], B["world"], f, d["x2"], d["c"], int(at >= B["meta"]["change_at"]))
            if not (ok1 and ok2):
                lost.add(("twin", name, f, k))
    for e in B["episodes"]:
        if e["kind"] == "first":
            n_first += 1
            if O.answer(e["f"], e["x"], e["c"], e["i"]) != e["o"]:       # the stream's reveal is the truth the oracle must match
                lost.add(("first", e["i"]))
    return lost, n_tw, n_first


def check2(B):
    lost, n_tw, n_first = oracle_scores(B, oracle_for(B))
    w = json.loads(json.dumps(B["world"])); w["u_new"] = {int(k): v for k, v in w["u_new"].items()}
    x0 = B["split"]["families"][0]["twins"][0][0][0]; w["v"][0][x0] = (w["v"][0][x0] + 1) % B["cfg"]["modulus"]      # one value wrong
    lost_fx, _, _ = oracle_scores(B, oracle_for(B, w))
    expect = {("twin", name, f, k) for name, f, items in panel_items(B) for k, d in enumerate(items) if f == 0 and x0 in (d["x1"], d["x2"])}
    expect |= {("first", e["i"]) for e in B["episodes"] if e["kind"] == "first" and e["f"] == 0 and e["x"] == x0}
    sep = {m: "sz_stream" not in (HERE / f"{m}.py").read_text().replace('"', "") and "import sz_" not in (HERE / f"{m}.py").read_text()
           for m in ("sz_oracle", "sz_solver")}
    return result(not lost and all(sep.values()), lost_fx == expect, twins_scored=n_tw, twins_lost=len([l for l in lost if l[0] == "twin"]),
                  first_encounter_scored=n_first, first_lost=len([l for l in lost if l[0] == "first"]), oracle_and_solver_import_no_generator=sep,
                  fixture=dict(what=f"v1({chr(65 + x0)}) off by one", lost=len(lost_fx), expected_exactly=len(expect), same_set=lost_fx == expect))


# ---------------------------------------------------------------- 3. visibility, by the prefix-only solver
def check3(B):
    WINDOW = B["cfg"]["window"]
    O = oracle_for(B); full_both = full_n = 0; suf = dict(both=0, single=0, abstain=0, members=0); fx_both = 0
    pairs = {}

    def fn(s, p, end):
        nonlocal full_both, full_n, fx_both
        if p["kind"] != "panel":
            return
        for f in (0, 1):
            for d in B["panels"][(p["name"], f)]:
                t1, t2 = O.answer(f, d["x1"], d["c"], p["at"]), O.answer(f, d["x2"], d["c"], p["at"])
                a = (s.answer_prompt(d["m1"]), s.answer_prompt(d["m2"])); full_n += 1
                ok = a == (t1, t2); full_both += ok
                key = (f, d["pair"]); pairs.setdefault(key, []).append(ok)
                b = (s.answer_prompt(d["m1"][-WINDOW:]), s.answer_prompt(d["m2"][-WINDOW:]))
                suf["both"] += b == (t1, t2); suf["members"] += 2
                suf["single"] += (b[0] == t1) + (b[1] == t2) - 2 * (b == (t1, t2)); suf["abstain"] += (b[0] is None) + (b[1] is None)
                c = (s.answer_prompt(d["m1"][:1] + d["m1"][-WINDOW:]), s.answer_prompt(d["m2"][:1] + d["m2"][-WINDOW:]))
                fx_both += c == (t1, t2)
    walk(B, "ordinary", fn)
    share = B["cfg"]["distinct_pair_share"]
    distinct = {f"family{f + 1}": sum(1 for (ff, _), v in pairs.items() if ff == f and sum(v) / len(v) >= share) for f in (0, 1)}
    return result(full_both == full_n and suf["both"] == 0, fx_both == full_n, twins=full_n, complete_episode_both_right=full_both,
                  distinct_twin_pairs_right_complete_episode=distinct, ten_byte_suffix=suf,
                  fixture=dict(what="suffix plus the context byte", both_right=fx_both, of=full_n))


# ---------------------------------------------------------------- 3b. memorisation tables
def tables_walk(B, extra_branches=None):
    """Three tables trained on the ordinary stream up to each probe: last ten bytes up to '=' -> outcome, last forty, exact (family,
    context, cue). extra_branches: {panel name: [(key40, (f, x, c), outcome)]}, added just before that panel (fixture)."""
    data = B["streams"]["ordinary"]; eps = parse(B["cfg"], data); t10, t40, tx = {}, {}, {}; M = B["cfg"]["modulus"]
    res = dict(twins=0, both={"10": 0, "40": 0, "exact": 0}, single={"10": 0, "40": 0, "exact": 0}, answered={"10": 0, "40": 0, "exact": 0},
               abstain={"10": 0, "40": 0, "exact": 0}, first=dict(n=0, right={"10": 0, "40": 0, "exact": 0}, answered={"10": 0, "40": 0, "exact": 0}))
    O = oracle_for(B); k = 0
    for p in sorted(B["probes"], key=lambda p: off(B, "ordinary", p["at"])):
        end = off(B, "ordinary", p["at"])
        while k < len(eps) and eps[k]["off"] < end:
            e = eps[k]; k += 1
            if e["type"] != "ep":
                continue
            t10[data[e["eq"] - 9:e["eq"] + 1]] = e["o"]; t40[data[max(0, e["eq"] - 39):e["eq"] + 1]] = e["o"]; tx[(e["f"], e["x"], e["c"])] = e["o"]
        if p["kind"] == "first":
            e = eps[k]; key = data[:e["eq"] + 1]; truth = e["o"]; res["first"]["n"] += 1
            for name, a in (("10", t10.get(key[-10:])), ("40", t40.get(key[-40:])), ("exact", tx.get((p["f"], p["x"], p["c"])))):
                res["first"]["answered"][name] += a is not None; res["first"]["right"][name] += a == truth
            continue
        for key40, fxc, o in (extra_branches or {}).get(p["name"], []):
            t40[key40] = o; tx[fxc] = o
        prefix = data[:end]
        for f in (0, 1):
            for d in B["panels"][(p["name"], f)]:
                res["twins"] += 1
                truth = (O.answer(f, d["x1"], d["c"], p["at"]), O.answer(f, d["x2"], d["c"], p["at"]))
                for name, look in (("10", lambda m, x: t10.get(m[-10:])), ("40", lambda m, x: t40.get((prefix + m)[-40:])),
                                   ("exact", lambda m, x: tx.get((f, x, d["c"])))):
                    a = (look(d["m1"], d["x1"]), look(d["m2"], d["x2"]))
                    both = a == truth; res["both"][name] += both
                    res["single"][name] += (a[0] == truth[0]) + (a[1] == truth[1]) - 2 * both
                    res["answered"][name] += (a[0] is not None) + (a[1] is not None); res["abstain"][name] += (a[0] is None) + (a[1] is None)
    res["expected_single_matches_by_chance"] = {n: round(v / M, 2) for n, v in res["answered"].items()}
    res["first"]["expected_by_chance"] = {n: round(v / M, 2) for n, v in res["first"]["answered"].items()}
    return res


def check3b(B):
    r = tables_walk(B)
    O = oracle_for(B); data = B["streams"]["ordinary"]; br = {}
    for p in B["probes"]:                  # fixture: each twin member inserted as its exact rendered prompt, after the bytes it has at its probe
        if p["kind"] == "panel":
            prefix = data[:off(B, "ordinary", p["at"])]
            br[p["name"]] = [((prefix + d[m])[-40:], (f, d[x], d["c"]), O.answer(f, d[x], d["c"], p["at"]))
                             for f in (0, 1) for d in B["panels"][(p["name"], f)] for m, x in (("m1", "x1"), ("m2", "x2"))]
    fx = tables_walk(B, br)
    ok = all(v == 0 for v in r["both"].values())
    return result(ok, fx["both"]["exact"] == fx["twins"] and fx["both"]["40"] == fx["twins"], **r,
                  fixture=dict(what="sealed twins inserted byte for byte after their probe's preceding bytes (one branch per member)",
                               twins=fx["twins"], both_right=fx["both"]))


# ---------------------------------------------------------------- 4. recency
def recency_walk(cfg, data, probe_list, truth_fn, obsolete_fn=None):
    """Answer = the last outcome seen in the prefix for the probe's cue. probe_list items: (offset, f, c, payload)."""
    eps = parse(cfg, data); last = {}; k = 0; out = []
    for o, f, c, payload in sorted(probe_list, key=lambda t: t[0]):
        while k < len(eps) and eps[k]["off"] < o:
            if eps[k]["type"] == "ep":
                last[(eps[k]["f"], eps[k]["c"])] = eps[k]["o"]
            k += 1
        out.append((last.get((f, c)), payload))
    return out


def check4(B):
    O = oracle_for(B); data = B["streams"]["ordinary"]
    tw = [(off(B, "ordinary", panel_at(B, n)), f, d["c"], (n, f, d)) for n, f, items in panel_items(B) for d in items]
    both = sum(a == O.answer(f, d["x1"], d["c"], panel_at(B, n)) and a == O.answer(f, d["x2"], d["c"], panel_at(B, n)) for a, (n, f, d) in recency_walk(B["cfg"], data, tw, None))
    single = sum((a == O.answer(f, d["x1"], d["c"], panel_at(B, n))) + (a == O.answer(f, d["x2"], d["c"], panel_at(B, n))) for a, (n, f, d) in recency_walk(B["cfg"], data, tw, None))
    rv = [(off(B, "ordinary", p["at"]), p["f"], p["c"], p) for p in B["probes"] if p["kind"] == "first" and p["set"] == "revise"]
    cnt = dict(changed_cue=dict(current=0, obsolete=0, neither=0), unchanged_cue=dict(current=0, neither=0))
    for a, p in recency_walk(B["cfg"], data, rv, None):
        cur = O.answer(p["f"], p["x"], p["c"], p["at"]); old = O.answer(p["f"], p["x"], p["c"], 0)
        if p["f"] == 0 and p["c"] in B["world"]["changed"]:
            cnt["changed_cue"]["current" if a == cur else "obsolete" if a == old else "neither"] += 1
        else:
            cnt["unchanged_cue"]["current" if a == cur else "neither"] += 1
    # fixture: one context only, where the last outcome for a cue is the answer
    cfg = B["cfg"]; w = B["world"]; r = G.rng(B["seed"], "fixture4"); eps = []
    for i in range(300):
        c = r.randrange(cfg["cues_per_family"]); eps.append(dict(f=0, x=0, c=c, fill=b"", o=G.true_outcome(cfg, w, 0, 0, c, 0)))
    fdata, fo = G.render(cfg, w, eps, None, "ordinary", B["seed"])
    seen, pl = set(), []
    for i, e in enumerate(eps):
        if (e["f"], e["c"]) in seen:
            pl.append((fo["starts"][i], 0, e["c"], e["o"]))
        seen.add((e["f"], e["c"]))
    fr = recency_walk(cfg, fdata, pl, None); full = sum(a == t for a, t in fr)
    return result(both == 0, full == len(fr), twins=len(tw), twins_both_right=both, single_members_right=single, revision_pairs=cnt,
                  fixture=dict(what="one context only", probes=len(fr), right=full))


# ---------------------------------------------------------------- 5. exposure ledger
def ledger(B, stream, episodes_meta=None):
    data = B["streams"][stream] if isinstance(stream, str) else stream
    toks = parse(B["cfg"], data); S = B["split"]["families"]
    sealed = {(f, x, c) for f in (0, 1) for tw in S[f]["twins"] for (x, c) in tw}
    held = {(f, x, c): s for f in (0, 1) for s in ("acquire", "retain", "revise") for (x, c) in S[f][s]}
    ctl = set(B["split"]["control_contexts"]) if (isinstance(stream, str) and stream.startswith("control")) else set()
    score_off = {}
    for p in B["probes"]:
        if p["kind"] == "first":
            score_off[(p["f"], p["x"], p["c"])] = off(B, stream, p["at"]) if isinstance(stream, str) else None
    counts, problems, line_names_pair = {}, [], 0
    for t in toks:
        key = (t["f"], t["x"], t["c"])
        if t["type"] == "line":                  # version 6: a line naming a pair is an occurrence of that pair's answer
            if key in sealed:
                problems.append(dict(line_names_sealed=key, off=t["off"]))
            if key in held:
                so = score_off.get(key)
                if ctl and t["f"] == 0 and t["x"] in ctl:
                    problems.append(dict(line_names_control_context_held_pair=key, off=t["off"]))
                elif so is None or t["off"] < so:
                    line_names_pair += 1; problems.append(dict(line_before_scoring=key, off=t["off"], scoring_at=so))
            continue
        counts[key] = counts.get(key, 0) + 1
        if key in sealed:
            problems.append(dict(sealed=key, off=t["off"]))
        if key in held:
            so = score_off.get(key)
            if ctl and t["f"] == 0 and t["x"] in ctl:
                problems.append(dict(control_context_held_pair=key, off=t["off"]))
            elif so is None or t["off"] < so:
                problems.append(dict(held_before_scoring=key, off=t["off"], scoring_at=so))
            elif counts[key] == 1 and t["off"] != so:
                problems.append(dict(first_occurrence_not_at_probe=key))
    return dict(occurrences=sum(counts.values()), distinct_pairs=len(counts), problems=problems[:10], n_problems=len(problems),
                lines_naming_exactly_a_held_pair_before_its_probe=line_names_pair)


def check5(B):
    out = {s: ledger(B, s) for s in B["streams"]}
    ok = all(v["n_problems"] == 0 for v in out.values())
    # fixture: one acquisition-held pair shown early as teaching
    e0 = next(e for e in B["episodes"] if e["kind"] == "first")
    eps = [dict(e) for e in B["episodes"]]; eps.insert(5, dict(e0, kind="teach"))
    fdata, fo = G.render(B["cfg"], B["world"], eps, None, "ordinary", B["seed"])
    B2 = dict(B, streams=dict(B["streams"], fixture=fdata), offsets=dict(B["offsets"], fixture=dict(fo, starts=fo["starts"][:5] + fo["starts"][6:])))
    fx = ledger(B2, "fixture")
    # fixture (version 6): one line, that of a first-encounter episode, placed before its pair's probe
    e1 = next(e for e in B["episodes"] if e["kind"] == "first" and e["i"] > 10)
    host = next(h for h in range(e1["i"] - 1, -1, -1) if B["episodes"][h]["segment"] == e1["segment"] and B["episodes"][h]["era"] == e1["era"])
    sl = sorted([dict(sv, host=host) if sv["describes"] == e1["i"] else sv for sv in B["slots"]], key=lambda sv: (sv["host"], sv["describes"]))
    ldata, lo = G.render(B["cfg"], B["world"], B["episodes"], sl, "lines", B["seed"])
    fx2 = ledger(dict(B, streams=dict(B["streams"], fixture_line=ldata), offsets=dict(B["offsets"], fixture_line=lo)), "fixture_line")
    return result(ok, fx["n_problems"] > 0 and fx2["n_problems"] > 0, **{s: dict(occurrences=v["occurrences"], problems=v["n_problems"],
                                                     lines_naming_exactly_a_held_pair_before_its_probe=v["lines_naming_exactly_a_held_pair_before_its_probe"]) for s, v in out.items()},
                  fixture=dict(what=f"held pair {chr(65 + e0['x'])}{chr(G.cue_byte(B["cfg"], e0['f'], e0['c']))} inserted as episode 5", reported=fx["n_problems"]),
                  fixture_line=dict(what=f"the line of first-encounter episode {e1['i']} ({chr(65 + e1['x'])}{chr(G.cue_byte(B["cfg"], e1['f'], e1['c']))}) hosted at episode {host}, before its probe",
                                    reported=fx2["n_problems"], problems=fx2["problems"][:3]))


# ---------------------------------------------------------------- 6. twins
def twin_problems(cfg, O, f, d, at):
    WINDOW, NCTX = cfg["window"], cfg["contexts"]
    p = []
    if d["m1"][-WINDOW:] != d["m2"][-WINDOW:]:
        p.append("last ten bytes differ")
    if d["x1"] == d["x2"] or d["m1"][0] == d["m2"][0]:
        p.append("same context")
    if len(d["m1"]) != len(d["m2"]):
        p.append("lengths differ")
    if O.answer(f, d["x1"], d["c"], at) == O.answer(f, d["x2"], d["c"], at):
        p.append("same outcome")
    if any(65 <= b < 65 + NCTX for b in d["m1"][-WINDOW:] + d["m2"][-WINDOW:]):
        p.append("context inside the window")
    return p


def check6(B):
    O = oracle_for(B); bad = []; n = 0
    distinct = {}
    for name, f, items in panel_items(B):
        for d in items:
            n += 1; pr = twin_problems(B["cfg"], O, f, d, panel_at(B, name))
            if pr:
                bad.append(dict(name=name, f=f, pair=d["pair"], problems=pr))
            distinct.setdefault(f"family{f + 1}", set()).add((d["x1"], d["x2"], d["c"]))
    w = json.loads(json.dumps(B["world"])); w["u_new"] = {int(k): v for k, v in w["u_new"].items()}
    d = B["panels"][("A", 0)][0]; w["v"][0][d["x2"]] = w["v"][0][d["x1"]]
    fx = twin_problems(B["cfg"], oracle_for(B, w), 0, d, panel_at(B, "A"))
    return result(not bad, "same outcome" in fx, renderings=n, rejected=bad[:5], distinct_twin_pairs={k: len(v) for k, v in distinct.items()},
                  twin_redraws=B["split"]["twin_redraws"], fixture=dict(what="a twin whose contexts share a value", rejected_for=fx))


# ---------------------------------------------------------------- 7. the moved-lines (alignment) and extra-bytes streams
def poisson_tail(k, lam):
    """P(X >= k) for X ~ Poisson(lam)."""
    term, total = math.exp(-lam), 0.0
    for i in range(k):
        total += term; term *= lam / (i + 1)
    return max(0.0, 1.0 - total)


def e_f(eps, s):
    return eps[s["describes"]]["f"]


def alignment(B, moved_slots):
    by_orig = {s["describes"]: s for s in B["slots"]}; eps = B["episodes"]; w = B["world"]
    fixed = same_content = 0; issues = []; classes = {}
    for s in moved_slots:
        o = by_orig[s["describes"]]                    # where the described episode's line stood in `wider`
        if (o["phase"], o["segment"], o["delay"], o["era"]) != (s["phase"], s["segment"], s["delay"], s["era"]):
            issues.append(dict(line=s["describes"], slot=s["host"], why="moved across phase, segment, delay stratum or value era"))
        if s["segment"] == "absent" and e_f(eps, s) == 0:
            issues.append(dict(line=s["describes"], slot=s["host"], why="a family-1 line where family 1 is absent"))
        host_era = eps[s["host"]]["era"]; e = eps[s["describes"]]
        true_now = G.line_bytes(B["cfg"], w, dict(e, era=host_era))
        if G.line_bytes(B["cfg"], w, e) != true_now:
            issues.append(dict(line=s["describes"], slot=s["host"], why="not true where it stands"))
        if not s.get("free", True):
            if s["describes"] != s["original"]:
                issues.append(dict(line=s["describes"], slot=s["host"], why="a first-encounter line moved"))
            continue
        classes.setdefault((s["segment"], s["delay"], s["era"]), []).append(s)
    for key, ss in classes.items():
        if len(ss) < 2:
            continue
        for s in ss:
            if s["describes"] == s["original"]:
                fixed += 1
            e, h = eps[s["describes"]], eps[s["original"]]
            same_content += (e["f"], e["x"], e["c"]) == (h["f"], h["x"], h["c"])
    lam = sum(1 for ss in classes.values() if len(ss) >= 2)          # a uniform permutation of each class fixes one line on average
    return dict(lines=len(moved_slots), lines_free_to_move=sum(len(v) for v in classes.values()),
                lines_moved=sum(1 for ss in classes.values() for s in ss if s["describes"] != s["original"]), aligned_lines=fixed, expected_by_chance=lam, p_value=round(poisson_tail(fixed, lam), 6),
                lines_describing_an_episode_of_the_same_pair_as_their_slot=same_content, issues=issues[:5], n_issues=len(issues))


def line_values(B, data, cb):
    """Every value read from the stream's bytes against the value allowed in the era in which the line is delivered."""
    w = B["world"]; bad, n = [], 0
    for t in parse(B["cfg"], data):
        if t["type"] != "line":
            continue
        n += 1; era = int(t["off"] >= cb)
        u_ok = w["u_new"][t["c"]] if (t["f"] == 0 and era == 1 and t["c"] in w["u_new"]) else w["u"][t["f"]][t["c"]]
        if t["vv"] != w["v"][t["f"]][t["x"]] or t["uu"] != u_ok:
            bad.append(dict(off=t["off"], line=f"{chr(65 + t['x'])}:{t['vv']:02d} {chr(G.cue_byte(B['cfg'], t['f'], t['c']))}:{t['uu']:02d}", era=era))
    return dict(lines=n, wrong=len(bad), first_wrong=bad[:3])


def solver_contradictions(cfg, data, cb):
    s = SV.PrefixSolver(cfg, cb); s.feed(data)
    return s.inconsistent


def check7(B):
    W, MV, X = B["streams"]["wider"], B["streams"]["moved"], B["streams"]["extra"]
    def names(data):
        out = {}
        for t in parse(B["cfg"], data):
            if t["type"] == "line":
                for k in ((t["f"], "x", t["x"]), (t["f"], "c", t["c"])):
                    out[k] = out.get(k, 0) + 1
        return out
    freq_same = names(W) == names(MV)
    al = alignment(B, B["moved_slots"])
    lr = B["offsets"]["wider"]["line_ranges"]; xr = B["offsets"]["extra"]["line_ranges"]
    placement = [(a, b) for a, b, *_ in lr] == [(a, b) for a, b, *_ in xr] and len(W) == len(X)
    outside = bytearray(W); inside_ok = True
    for a, b, *_ in lr:
        inside_ok &= all(ch in G.FILL for ch in X[a:b]); outside[a:b] = X[a:b]
    placement = placement and bytes(outside) == X and inside_ok
    vals = {s: line_values(B, B["streams"][s], change_byte(B, s)) for s in ("wider", "moved", "control_wider")}
    contra = {s: solver_contradictions(B["cfg"], B["streams"][s], change_byte(B, s)) for s in ("ordinary", "wider", "moved", "extra", "control", "control_wider")}
    ok = (freq_same and al["n_issues"] == 0 and al["p_value"] > 0.001 and placement and all(v["wrong"] == 0 for v in vals.values())
          and all(v == 0 for v in contra.values()))
    fx_slots, _ = G.move_lines(B["slots"], B["seed"], B["episodes"], in_place=True)
    fx = alignment(B, fx_slots)
    # fixture (version 6): one wrong value in one line of the moved stream
    a, b, *_ = next(r_ for r_ in B["offsets"]["moved"]["line_ranges"] if r_[0] > change_byte(B, "moved"))
    bad = bytearray(MV); vv = int(bad[a + 2:a + 4]); bad[a + 2:a + 4] = b"%02d" % ((vv + 1) % B["cfg"]["modulus"]); bad = bytes(bad)
    fxv = dict(line=bad[a:b - 1].decode(), was=MV[a:b - 1].decode(), values=line_values(B, bad, change_byte(B, "moved")),
               solver_contradictions=solver_contradictions(B["cfg"], bad, change_byte(B, "moved")))
    fxv["failed"] = fxv["values"]["wrong"] == 1 and fxv["solver_contradictions"] > 0
    return result(ok, fx["p_value"] <= 0.001 and fxv["failed"], line_values_from_bytes=vals, solver_contradicting_rows=contra,
                  fixture_wrong_value=fxv, symbol_frequencies_equal=freq_same, **al, lines_that_stayed=B["move_info"]["lines_that_stayed"],
                  move_classes=B["move_info"]["classes"], lines_fixed_first_encounter=B["move_info"]["lines_fixed_first_encounter"],
                  extra_bytes_match_count_and_placement=placement,
                  disclosure_ledger=disclosure_ledger(B),
                  fixture=dict(what="a shuffle that leaves every line in place", aligned_lines=fx["aligned_lines"], p_value=fx["p_value"]))


def disclosure_ledger(B):
    """For every probe and stream: which context and cue values a line has disclosed so far; the arms compared probe by probe.
    (Sizing copy: the same sets as step 0's, accumulated in one sweep over each stream's lines instead of rescanned at every probe.)"""
    streams = ("ordinary", "wider", "moved", "extra")
    lines = {s: [t for t in parse(B["cfg"], B["streams"][s]) if t["type"] == "line"] for s in streams}
    rows = []; differ = 0; probed_known = {s: 0 for s in streams}; exact_pair_line = {s: 0 for s in streams}
    acc = {s: dict(k=0, ctx=set(), cue=set(), pair=set()) for s in streams}
    for p in sorted(B["probes"], key=lambda p: p["at"]):
        row = dict(at=p["at"], kind=p["kind"])
        sets = {}
        for s in streams:
            end = off(B, s, p["at"]); cb = change_byte(B, s); a = acc[s]; L = lines[s]
            while a["k"] < len(L) and L[a["k"]]["off"] < end:
                t = L[a["k"]]; a["k"] += 1
                a["ctx"].add((t["f"], t["x"])); a["cue"].add((t["f"], t["c"], int(t["off"] >= cb))); a["pair"].add((t["f"], t["x"], t["c"]))
            ctx, cue = a["ctx"], a["cue"]
            sets[s] = (set(ctx), set(cue)); row[s] = [len(ctx), len(cue)]
            if p["kind"] == "first":
                era = int(p["at"] >= B["meta"]["change_at"])
                if (p["f"], p["x"]) in ctx and (p["f"], p["c"], era) in cue:
                    probed_known[s] += 1
                exact_pair_line[s] += (p["f"], p["x"], p["c"]) in a["pair"]
        differ += sets["wider"] != sets["moved"]
        rows.append(row)
    return dict(probes=len(rows), probes_where_wider_and_moved_had_disclosed_different_values=differ,
                first_encounter_probes_with_both_values_already_disclosed=probed_known,
                first_encounter_probes_preceded_by_a_line_naming_exactly_that_pair=exact_pair_line,
                per_probe_counts_ctx_cue=rows)


# ---------------------------------------------------------------- 8. determinism and hashes
def check8(B, other_seed):
    B2 = G.build(B["cfg"], B["seed"]); same = all(B["streams"][k] == B2["streams"][k] for k in B["streams"]) and G.hashes(B) == G.hashes(B2)
    h, ho = G.hashes(B), G.hashes(G.build(B["cfg"], other_seed))
    flat = lambda d: {f"{a}/{b}": v for a, sub in d.items() for b, v in sub.items()}
    unchanged = [k for k, v in flat(h).items() if flat(ho)[k] == v]
    return result(same, not unchanged, hashes=h, fixture=dict(what=f"seed {other_seed} instead", hashes_unchanged=unchanged))


# ---------------------------------------------------------------- 9. the prior-teaching control
def control_undetermined(B, stream, shared=False):
    ctl = B["split"]["control_contexts"]; det = []

    def fn(s, p, end):
        for x in ctl:
            for c in range(B["cfg"]["cues_per_family"]):
                a = s.query(0, x, c)
                if a is not None:
                    det.append(dict(at=p["at"], x=chr(65 + x), c=chr(97 + c), answer=a))
    walk(B, stream, fn, shared)
    return det


def check9(B):
    det = {s: control_undetermined(B, s) for s in ("control", "control_wider")}
    eps, ceps = B["episodes"], B["control_episodes"]; ctl = set(B["split"]["control_contexts"])
    taught0 = set(B["split"]["families"][0]["taught"]); bad = []
    for e, n in zip(eps, ceps):
        if (e["f"], e["c"], e["fill"], e["phase"], e["era"]) != (n["f"], n["c"], n["fill"], n["phase"], n["era"]):
            bad.append(dict(at=e["i"], why="timing, cue or filler differ"))
        if e["f"] == 0 and e["x"] in ctl and ((n["x"], n["c"]) not in taught0 or n["x"] in ctl):
            bad.append(dict(at=e["i"], why="replacement not a taught pair of a non-control context"))
        if (e["f"] != 0 or e["x"] not in ctl) and e != {k: v for k, v in n.items()}:
            bad.append(dict(at=e["i"], why="an episode outside the control changed"))
    leaks = [t for s in ("control", "control_wider") for t in parse(B["cfg"], B["streams"][s]) if t["f"] == 0 and t["x"] in ctl]
    ok = not det["control"] and not det["control_wider"] and not bad and not leaks and len(eps) == len(ceps)
    # fixture: a world in which the two families share context values
    BF = G.build(B["cfg"], B["seed"], shared_context_values=True)
    fx = control_undetermined(BF, "control", shared=True)
    n_first = sum(1 for e in eps if e["kind"] == "first" and e["f"] == 0 and e["x"] in ctl)
    return result(ok, bool(fx), control_contexts=[chr(65 + x) for x in sorted(ctl)], episodes_replaced=len(B["control_replaced"]),
                  of_which_first_encounter_reveals=n_first, determined_answers=dict(control=len(det["control"]), control_wider=len(det["control_wider"])),
                  matching_problems=bad[:5], family1_bytes_naming_a_control_context=len(leaks),
                  fixture=dict(what="the families share context values", determined_answers=len(fx)))


# ---------------------------------------------------------------- 10. probes leave no trace
class StatefulResponder:
    """A fixture learner: its state is a digest of every byte it has read; it answers from that state."""
    def __init__(self, cfg):
        self.M = cfg["modulus"]; self.state = G.sha(b""); self.read_from = []

    def get_state(self):
        return self.state.encode()

    def set_state(self, s):
        self.state = s.decode()

    def feed(self, data):
        self.state = G.sha(self.state.encode() + data)

    def answer(self, prompt):
        self.read_from.append(self.state); self.feed(prompt); return int(self.state[:8], 16) % self.M


def bad_score_twin(resp, m1, m2):
    return resp.answer(m1), resp.answer(m2)


def no_trace(B, scorer):
    R = StatefulResponder(B["cfg"]); data = B["streams"]["ordinary"]; pos = 0; bad = 0; n = 0
    for p in sorted(B["probes"], key=lambda p: p["at"]):
        if p["kind"] != "panel":
            continue
        e = off(B, "ordinary", p["at"]); R.feed(data[pos:e]); pos = e
        for f in (0, 1):
            for d in B["panels"][(p["name"], f)]:
                before = R.get_state(); R.read_from = []
                scorer(R, d["m1"], d["m2"]); n += 1
                if R.get_state() != before or len(set(R.read_from)) != 1 or R.read_from[0] != before.decode():
                    bad += 1
                    R.set_state(before)
    return n, bad


def check10(B):
    n, bad = no_trace(B, G.score_twin)
    _, fbad = no_trace(B, bad_score_twin)
    return result(bad == 0, fbad > 0, twins=n, twins_with_a_trace=bad, fixture=dict(what="a scorer that does not restore", twins_with_a_trace=fbad))


# ---------------------------------------------------------------- run
def counts(B):
    S = B["split"]["families"]; eps = B["episodes"]
    return dict(episodes=len(eps), segments=B["meta"]["segments"], change_at_episode=B["meta"]["change_at"],
                change_at_byte={s: change_byte(B, s) for s in B["streams"]}, panel_probes=B["meta"]["panel_probes"],
                bytes={k: len(v) for k, v in B["streams"].items()}, lines=len(B["slots"]),
                per_family={f"family{f + 1}": dict(taught=len(S[f]["taught"]), acquire=len(S[f]["acquire"]), retain=len(S[f]["retain"]),
                                                   revise=len(S[f]["revise"]), sealed_twin_pairs=len(S[f]["twins"])) for f in (0, 1)},
                changed_cues=[chr(97 + c) for c in B["world"]["changed"]], twin_redraws=B["split"]["twin_redraws"],
                split_redraws=B["split"]["split_redraws"], revision_placement_attempts=B["meta"]["placement_attempts"],
                renderings_per_panel=B["cfg"]["renderings_per_panel"])


def run_checks(B, other_seed):
    """All checks of step 0 for one bundle (one configuration, one world seed). Returns {check: result}."""
    return {"1": check1(B), "2": check2(B), "3": check3(B), "3b": check3b(B), "4": check4(B), "5": check5(B), "6": check6(B),
            "7": check7(B), "8": check8(B, other_seed), "9": check9(B), "10": check10(B)}


def parity(default_cfg):
    """Step 0's record reproduced from the copies at the default configuration, for seeds 1, 2, 3."""
    out = dict(brief="OPUS_TASK_TARGET_TEST_STEP0_20261111.md (version 5)", sizes=default_cfg, seeds={}, note="No learner is trained; nothing is concluded about any learner.")
    failed = []
    for i, seed in enumerate(SEEDS):
        B = G.build(default_cfg, seed); other = SEEDS[(i + 1) % len(SEEDS)]
        checks = run_checks(B, other)
        out["seeds"][str(seed)] = dict(counts=counts(B), checks=checks)
        for k, v in checks.items():
            if not v["passed"]:
                failed.append((seed, k))
    out["passed"] = not failed; out["failed"] = failed
    return out


def compare(a, b, path="", renamed=None, diffs=None):
    """Every value of a against b; renamed maps a key of a to the key it must equal in b."""
    diffs = [] if diffs is None else diffs; renamed = renamed or {}
    if isinstance(a, dict) and isinstance(b, dict):
        ka = {renamed.get(k, k): k for k in a}
        for k in sorted(set(ka) | set(b)):
            if k not in ka or k not in b:
                diffs.append(dict(path=f"{path}/{k}", why="key only on one side")); continue
            compare(a[ka[k]], b[k], f"{path}/{k}", renamed, diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(dict(path=path, why=f"length {len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", renamed, diffs)
    elif a != b or type(a) is not type(b):
        diffs.append(dict(path=path, ours=str(a)[:80], on_file=str(b)[:80]))
    return diffs


def main():
    import sz_config as CF
    if sys.argv[1:2] != ["parity"]:
        print(__doc__); sys.exit(2)
    out = parity(CF.default())
    mine = json.loads(json.dumps(out, indent=1, default=str))
    (HERE / "SZ_PARITY_CHECK.json").write_text(json.dumps(mine, indent=1))
    filed = json.loads((HERE.parent / "TARGET_STEP0_CHECK.json").read_text())
    renamed = {"sz_oracle": "target_oracle", "sz_solver": "target_solver"}       # check 2 names the files it read; the only allowed difference
    diffs = compare(mine, filed, renamed=renamed)
    leaves = sum(1 for _ in _leaves(filed))
    print(f"parity: checks passed={out['passed']}  values on file={leaves}  differences from TARGET_STEP0_CHECK.json: {len(diffs)}")
    for d in diffs[:20]:
        print("  ", d)
    sys.exit(0 if (out["passed"] and not diffs) else 1)


def _leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _leaves(v)
    elif isinstance(x, list):
        for v in x:
            yield from _leaves(v)
    else:
        yield x


if __name__ == "__main__":
    main()
