"""Sizing sweep: the probe schedule, the readings, the references and the five classes (brief rev. 3, "What is read" and "What the table
is for"). Shared by every fit and by the admission fixtures, so a fixture responder is scored by exactly the code that scores the learner.

Arms: "A" (episode by episode, positions are episode indices, ordinary stream only) and "B" (one continuing stream, positions are target
byte indices). A responder has advance_to(position) and respond(prompts) -> list of bytes (see sz_learner).

Events, in order (all positions are boundaries in the brief's sense, set by the last loss target used):
    first   every first-encounter episode: arm B at its '=' + 1 (the chunk before ends with '=' as its last target; its pending input is '='),
            prompt b""; arm A at its episode index, prompt the episode's bytes up to '='.
    panel   A (end of acquire), I (family 1's return), R (end of stream), at the start of the episode at that index (arm B: its first byte;
            R: the end), with the sealed twins of step 0 (both members answered from one state saved before either); at A and R also the
            frozen taught panel (sz_panels.py).
References, from each arm's own prefix (learner-free): recitation shortcuts (ten-byte-suffix table, exact-key table, recency table) and,
in streams with lines, the direct-use benchmark; the exact-key table is the taught panel's ceiling, not a shortcut to beat there.
"""
import bisect

import sz_check as CK
import sz_oracle as OR
import sz_stream as G

SUFFIX = 10


def direct_use_answer(cfg, f, v, u):
    """The direct-use benchmark's arithmetic: the family's operation on the two stated values, modulo the configuration's modulus."""
    M = cfg["modulus"]
    return (v + u) % M if f == 0 else (v - u) % M


def truth_bytes(a):
    return b"%02d\n" % a


# ---------------------------------------------------------------- the schedule
def plan(B, stream, arm, taught):
    """Events and the cut positions they impose. taught: the frozen taught-panel items of this world and modulus."""
    assert arm in ("A", "B") and (arm == "B" or stream == "ordinary")
    cfg = B["cfg"]; eps = B["episodes"]; off = B["offsets"][stream]
    O = OR.Oracle(cfg, B["world"]["v"], B["world"]["u"], B["world"]["u_new"], B["meta"]["change_at"])
    ev = []
    for p in B["probes"]:
        if p["kind"] == "first":
            e = eps[p["at"]]
            pos = off["eqs"][p["at"]] + 1 if arm == "B" else p["at"]
            prompt = b"" if arm == "B" else G.episode_bytes(cfg, e, outcome=False)
            truth = O.answer(p["f"], p["x"], p["c"], p["at"])
            assert truth == e["o"]
            ev.append(dict(type="first", pos=pos, at=p["at"], probe=p, prompts=[prompt], truth=[truth]))
        else:
            at = p["at"]
            pos = (off["starts"][at] if at < len(eps) else off["end"]) if arm == "B" else at
            prompts, truth, meta = [], [], []
            for f in (0, 1):
                for k, d in enumerate(B["panels"][(p["name"], f)]):
                    for m, x in (("m1", "x1"), ("m2", "x2")):
                        prompts.append(d[m]); truth.append(O.answer(f, d[x], d["c"], at))
                        meta.append(dict(kind="twin", f=f, k=k, pair=d["pair"], member=m, x=d[x], c=d["c"]))
            if p["name"] in ("A", "R"):
                for it in taught:
                    prompts.append(it["prompt"]); truth.append(O.answer(it["f"], it["x"], it["c"], at))
                    assert truth[-1] == it["truth"]
                    meta.append(dict(kind="taught", f=it["f"], x=it["x"], c=it["c"], filler=it["filler"]))
            ev.append(dict(type="panel", name=p["name"], pos=pos, at=at, prompts=prompts, truth=truth, meta=meta))
    ev.sort(key=lambda e: (e["pos"], e["type"] == "first"))
    end = off["end"] if arm == "B" else len(eps)
    return ev, sorted({e["pos"] for e in ev if 0 < e["pos"] < end}), end


# ---------------------------------------------------------------- running a responder through the schedule
def run(responder, ev, end, probes_on=True):
    """Advance to each event's boundary, answer its prompts (or skip them), and train to the end. Returns the responses per event."""
    out = []
    for e in ev:
        responder.advance_to(e["pos"])
        out.append(responder.respond(e["prompts"]) if probes_on else None)
    responder.advance_to(end)
    return out


# ---------------------------------------------------------------- references, learner-free, from the arm's own prefix
def references(B, stream, arm, ev, grid=None):
    """For every event and prompt: the three recitation shortcuts, the direct-use benchmark (lines streams), the exposure counts and, for
    each use of a line, its distance in bytes to the probe and the cut positions between (grid: the learner's chunk boundaries)."""
    cfg = B["cfg"]
    src = stream if arm == "B" else "ordinary"
    data = B["streams"][src]; off = B["offsets"][src]
    toks = CK.parse(cfg, data)
    cb = off["starts"][B["meta"]["change_at"]]
    t10, tx, rec = {}, {}, {}
    exp_pair, exp_ctx, exp_cue, line_ctx_n, line_cue_n = {}, {}, {}, {}, {}
    last_v, last_u = {}, {}
    k = 0; out = []
    for e in ev:
        lim = e["pos"] if arm == "B" else (off["starts"][e["pos"]] if e["pos"] < len(off["starts"]) else off["end"])
        while k < len(toks):
            t = toks[k]
            tend = (t["eq"] + 3) if t["type"] == "ep" else (t["off"] + 9)        # the token's last byte (an episode's end byte, a line's)
            if tend >= lim:
                break
            k += 1
            if t["type"] == "ep":
                fx, fc, pr = (t["f"], t["x"]), (t["f"], t["c"]), (t["f"], t["x"], t["c"])
                if arm == "B":
                    key = data[max(0, t["eq"] - SUFFIX + 1):t["eq"] + 1]
                else:
                    key = data[max(t["off"], t["eq"] - SUFFIX + 1):t["eq"] + 1]
                t10[key] = t["o"]; tx[pr] = t["o"]; rec[fc] = t["o"]
                exp_pair[pr] = exp_pair.get(pr, 0) + 1; exp_ctx[fx] = exp_ctx.get(fx, 0) + 1; exp_cue[fc] = exp_cue.get(fc, 0) + 1
            else:
                era = int(t["off"] >= cb)
                last_v[(t["f"], t["x"])] = (t["vv"], t["off"] + 10); last_u[(t["f"], t["c"], era)] = (t["uu"], t["off"] + 10)
                line_ctx_n[(t["f"], t["x"])] = line_ctx_n.get((t["f"], t["x"]), 0) + 1
                line_cue_n[(t["f"], t["c"])] = line_cue_n.get((t["f"], t["c"]), 0) + 1
        era = int(e["at"] >= B["meta"]["change_at"])
        rows = []
        items = ([dict(f=e["probe"]["f"], x=e["probe"]["x"], c=e["probe"]["c"])] if e["type"] == "first" else e["meta"])
        for i, it in enumerate(items):
            f, x, c = it["f"], it["x"], it["c"]
            if e["type"] == "first":
                q = off["eqs"][e["at"]]
                key = data[max(0, q - SUFFIX + 1):q + 1] if arm == "B" else data[max(off["starts"][e["at"]], q - SUFFIX + 1):q + 1]
            else:
                key = (data[:lim] + e["prompts"][i])[-SUFFIX:] if arm == "B" else e["prompts"][i][-SUFFIX:]
            du = None; dist = None
            if (f, x) in last_v and (f, c, era) in last_u:
                (vv, vend), (uu, uend) = last_v[(f, x)], last_u[(f, c, era)]
                du = direct_use_answer(cfg, f, vv, uu)
                if grid is not None:
                    d = {}
                    for name, lend in (("context_line", vend), ("cue_line", uend)):
                        a = bisect.bisect_right(grid, lend); b = bisect.bisect_right(grid, lim)     # cuts g with lend < g <= lim
                        d[name] = dict(bytes=lim - lend, cuts_between=max(0, b - a))
                    dist = d
            rows.append(dict(suffix10=t10.get(key), exact=tx.get((f, x, c)), recency=rec.get((f, c)), direct_use=du, line_distance=dist,
                             exposure=dict(pair=exp_pair.get((f, x, c), 0), context=exp_ctx.get((f, x), 0), cue=exp_cue.get((f, c), 0),
                                           context_in_lines=line_ctx_n.get((f, x), 0), cue_in_lines=line_cue_n.get((f, c), 0))))
        out.append(rows)
    return out


# ---------------------------------------------------------------- readings
SHORTCUTS = ("suffix10", "exact", "recency")


def readings(B, ev, responses, refs, lines_stream):
    cfg = B["cfg"]; share = cfg["distinct_pair_share"]
    first, twins, taught = [], {}, {}
    for e, resp, ref in zip(ev, responses, refs):
        if e["type"] == "first":
            p = e["probe"]; t = e["truth"][0]; r = ref[0]
            rec = dict(at=p["at"], set=p["set"], phase=p["phase"], era=p["era"], family=p["f"] + 1, x=p["x"], c=p["c"], truth=t,
                       response=resp[0].decode("latin-1"), right=resp[0] == truth_bytes(t), exposure=r["exposure"],
                       shortcuts={s: r[s] == t for s in SHORTCUTS}, shortcut_answered={s: r[s] is not None for s in SHORTCUTS})
            if lines_stream:
                rec["direct_use"] = r["direct_use"] == t; rec["line_distance"] = r["line_distance"]
            first.append(rec)
            continue
        tw = {}; tp = {}
        for i, (m, t, rsp, r) in enumerate(zip(e["meta"], e["truth"], resp, ref)):
            if m["kind"] == "twin":
                tw.setdefault((m["f"], m["k"]), {})[m["member"]] = dict(pair=m["pair"], right=rsp == truth_bytes(t),
                                                                       sc={s: r[s] == t for s in SHORTCUTS}, du=r["direct_use"] == t,
                                                                       exposure=r["exposure"], dist=r["line_distance"])
            else:
                tp.setdefault((m["f"], m["filler"]), []).append(dict(right=rsp == truth_bytes(t), ceiling=r["exact"] == t))
        T = {}
        for f in (0, 1):
            rend = [v for (ff, _), v in tw.items() if ff == f]
            both = lambda pick: [pick(v["m1"]) and pick(v["m2"]) for v in rend]
            per_pair = {}
            for v, b in zip(rend, both(lambda z: z["right"])):
                per_pair.setdefault(v["m1"]["pair"], []).append(b)
            fam = dict(renderings=len(rend), both_right=sum(both(lambda z: z["right"])),
                       members_right=sum(v["m1"]["right"] + v["m2"]["right"] for v in rend),
                       distinct_pairs=len(per_pair), distinct_pairs_right=sum(1 for v in per_pair.values() if sum(v) / len(v) >= share),
                       shortcuts_both_right={s: sum(both(lambda z, s=s: z["sc"][s])) for s in SHORTCUTS},
                       exposure_members_mean=dict(context=sum(v[m]["exposure"]["context"] for v in rend for m in ("m1", "m2")) / max(1, 2 * len(rend)),
                                                  cue=sum(v["m1"]["exposure"]["cue"] for v in rend) / max(1, len(rend)),
                                                  pair=sum(v[m]["exposure"]["pair"] for v in rend for m in ("m1", "m2"))))
            if lines_stream:
                fam["direct_use_both_right"] = sum(both(lambda z: z["du"]))
            T[f"family{f + 1}"] = fam
        twins[e["name"]] = T
        if tp:
            taught[e["name"]] = {f"family{f + 1}/{k}": dict(n=len(v), right=sum(z["right"] for z in v), exact_key_ceiling=sum(z["ceiling"] for z in v))
                                 for (f, k), v in sorted(tp.items())}
    summ = {}
    for s in ("acquire", "retain", "revise"):
        sub = [r for r in first if r["set"] == s]
        summ[s] = dict(n=len(sub), right=sum(r["right"] for r in sub), **{f"shortcut_{k}": sum(r["shortcuts"][k] for r in sub) for k in SHORTCUTS},
                       by_family={f"family{f}": sum(r["right"] for r in sub if r["family"] == f) for f in (1, 2)})
        if lines_stream:
            summ[s]["direct_use"] = sum(r["direct_use"] for r in sub)
    return dict(first_encounter=first, first_encounter_summary=summ, twins=twins, taught=taught)


def classify(R):
    """The five classes, fixed before any fit (brief rev. 3). R: readings of one completed, admitted fit. Returns (class, the numbers)."""
    acq = R["first_encounter_summary"]["acquire"]
    acc = acq["right"] / acq["n"]
    best = max(acq[f"shortcut_{k}"] for k in SHORTCUTS) / acq["n"]
    A = R["twins"]["A"]
    dist = sum(A[f]["distinct_pairs"] for f in A); dist_right = sum(A[f]["distinct_pairs_right"] for f in A)
    held = acc >= best + 0.20 and dist_right >= 0.20 * dist
    tA = R["taught"]["A"]
    long_n = sum(v["n"] for k, v in tA.items() if k.endswith("/long")); long_r = sum(v["right"] for k, v in tA.items() if k.endswith("/long"))
    nums = dict(acquire_first_encounter_accuracy=acc, best_recitation_shortcut=best, margin=acc - best, distinct_twin_pairs_right=dist_right,
                distinct_twin_pairs=dist, taught_long_recall_at_A=long_r / long_n)
    if held:
        return "held success", nums
    if long_r / long_n >= 0.5:
        return "taught-only success", nums
    return "negative", nums


def setting_class(classes):
    """A setting counts for a decision only if its three worlds agree."""
    return classes[0] if len(set(classes)) == 1 else "mixed"


# ---------------------------------------------------------------- fixture responders (admission)
class _Fixture:
    def __init__(self, B, stream, arm):
        self.B, self.arm = B, arm; self.cfg = B["cfg"]
        self.data = B["streams"][stream if arm == "B" else "ordinary"]; self.off = B["offsets"][stream if arm == "B" else "ordinary"]
        self.pos = 0

    def advance_to(self, b):
        assert b >= self.pos
        self.pos = b

    def prefix(self):
        return self.data[:self.pos] if self.arm == "B" else b""

    def episode_index(self):
        if self.arm == "A":
            return self.pos
        return bisect.bisect_right(self.off["starts"], self.pos) - 1 if self.pos < self.off["end"] else len(self.B["episodes"])


class OracleResponder(_Fixture):
    """Answers every prompt from the hidden values: the context and cue read from the prompt's own episode text."""

    def __init__(self, B, stream, arm):
        super().__init__(B, stream, arm)
        w = B["world"]; self.O = OR.Oracle(self.cfg, w["v"], w["u"], w["u_new"], B["meta"]["change_at"])

    def respond(self, prompts):
        out = []; at = self.episode_index(); K = self.cfg["cues_per_family"]
        tail = self.prefix()[-64:]
        for p in prompts:
            text = (tail + p).rsplit(b"\n", 1)[-1].lstrip(G.FILL)
            x = text[0] - 65; cb = text[-2]; f, c = (cb - 97) // K, (cb - 97) % K
            out.append(truth_bytes(self.O.answer(f, x, c, at)))
        return out


class SuffixResponder(_Fixture):
    """Answers from the last ten bytes before '=' alone: a table of every episode revealed in the prefix (abstains as b"")."""

    def __init__(self, B, stream, arm):
        super().__init__(B, stream, arm)
        self.toks = [t for t in CK.parse(self.cfg, self.data) if t["type"] == "ep"]; self.k = 0; self.table = {}

    def respond(self, prompts):
        lim = self.pos if self.arm == "B" else (self.off["starts"][self.pos] if self.pos < len(self.off["starts"]) else self.off["end"])
        while self.k < len(self.toks) and self.toks[self.k]["eq"] + 3 < lim:
            t = self.toks[self.k]; self.k += 1
            lo = max(0, t["eq"] - SUFFIX + 1) if self.arm == "B" else max(t["off"], t["eq"] - SUFFIX + 1)
            self.table[self.data[lo:t["eq"] + 1]] = t["o"]
        out = []
        for p in prompts:
            key = (self.data[max(0, self.pos - SUFFIX):self.pos] + p)[-SUFFIX:] if self.arm == "B" else p[-SUFFIX:]
            a = self.table.get(key)
            out.append(truth_bytes(a) if a is not None else b"")
        return out
