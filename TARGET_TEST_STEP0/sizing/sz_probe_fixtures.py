"""Sizing sweep, stage 1: the probe protocol shown on the real network and training path (brief rev. 3, "A probe inside the stream").

World 104 (profiling only, in no roster), default configuration (modulus 97, four presentations), model seed 1104. Nothing here is a fit;
every learner is discarded. Each fixture states the condition that must hold and the broken variant that must fail it.

  F1 reveal invariance: the bytes of the reveal and everything after it changed -> the response to the probe unchanged, bitwise in its
     logits (arm B: the first first-encounter probe and the twins and taught panel at panel A; arm A: the first first-encounter probe).
  F2 the boundary rule can fail: a chunk (arm B) or batch (arm A) that has used the first answer digit as a target before the probe. With
     the first revealed digit set in turn to 0..9, the broken boundary's response follows the reveal; the correct boundary's does not move.
  F3 twins: the second member gives the same answer, bitwise in its logits, whether or not the first was presented before it; batched and
     one-by-one answers agree on every twin prompt of the panel; a broken protocol that answers the second member from the state the first
     member left behind changes it.
  F4 probe-on and probe-off runs agree afterwards in weights, optimizer state, carried state, cursor and every counter (both arms, through
     panel A and beyond); a leaky probe that keeps the state it reached does not.
"""
import copy
import random
import sys
import time
from pathlib import Path

import torch as T

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sz_config as CF     # noqa: E402
import sz_learner as L     # noqa: E402
import sz_panels as P      # noqa: E402
import sz_score as SC      # noqa: E402
import sz_stream as G      # noqa: E402

WORLD = CF.PROFILE_WORLD
MODEL_SEED = WORLD + 1000


def setup():
    cfg = CF.make(97, 4)
    B = G.build(cfg, WORLD, streams_wanted=("ordinary",))
    taught, _ = P.load(WORLD, 97)
    return cfg, B, taught


def mutate(data, start, seed):
    """Every byte from `start` on replaced: digits by another digit, every other byte by another byte of the same kind of symbol."""
    r = random.Random(seed); out = bytearray(data)
    for k in range(start, len(out)):
        b = out[k]
        if 48 <= b <= 57:
            out[k] = 48 + (b - 48 + r.randrange(1, 10)) % 10
        elif b in G.FILL:
            out[k] = r.choice([c for c in G.FILL if c != b])
        elif 65 <= b < 91:
            out[k] = 65 + (b - 65 + r.randrange(1, 24)) % 24
        elif 97 <= b < 121:
            out[k] = 97 + (b - 97 + r.randrange(1, 24)) % 24
    return bytes(out)


def same_tensors(a, b):
    return all(T.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()


def diff_keys(a, b):
    return [k for k in a if k not in b or not T.equal(a[k], b[k])]


def stream_learner(B, data, cuts):
    return L.StreamLearner(MODEL_SEED, data, cuts)


def episode_list(B, eps=None):
    return [G.episode_bytes(B["cfg"], e) for e in (eps or B["episodes"])]


# ---------------------------------------------------------------- F1
def f1(cfg, B, taught):
    s = B["streams"]["ordinary"]
    ev, cuts, end = SC.plan(B, "ordinary", "B", taught)
    first = next(e for e in ev if e["type"] == "first"); q1 = first["pos"]
    panelA = next(e for e in ev if e["type"] == "panel" and e["name"] == "A"); pA = panelA["pos"]
    out = {}
    # arm B, first encounter: everything from the first answer byte on is changed
    s2 = mutate(s, q1, 1)
    la, lb = stream_learner(B, s, cuts), stream_learner(B, s2, cuts)
    la.advance_to(q1); lb.advance_to(q1)
    ra, ga = la.respond_logits(b""); rb, gb = lb.respond_logits(b"")
    out["B_first"] = dict(probe_at_episode=first["at"], boundary=q1, bytes_changed=sum(x != y for x, y in zip(s, s2)), updates=la.updates,
                          response=[ra.decode("latin-1"), rb.decode("latin-1")], logits_bitwise_equal=bool(T.equal(ga, gb)),
                          batched_responses_equal=la.respond(first["prompts"]) == lb.respond(first["prompts"]))
    # arm B, panel A: everything from the panel's place on is changed
    s3 = mutate(s, pA, 2)
    la, lb = stream_learner(B, s, cuts), stream_learner(B, s3, cuts)
    la.advance_to(pA); lb.advance_to(pA)
    resp_a, resp_b = la.respond(panelA["prompts"]), lb.respond(panelA["prompts"])
    lg_equal = all(T.equal(la.respond_logits(p)[1], lb.respond_logits(p)[1]) for p in panelA["prompts"][:40])
    out["B_panelA"] = dict(boundary=pA, prompts=len(panelA["prompts"]), bytes_changed=sum(x != y for x, y in zip(s, s3)),
                           responses_equal=resp_a == resp_b, logits_bitwise_equal_first_40=lg_equal, updates=la.updates)
    keepB = la
    # arm A, first encounter: the probe's reveal and every later episode changed
    evA, cutsA, endA = SC.plan(B, "ordinary", "A", taught)
    fA = next(e for e in evA if e["type"] == "first"); i1 = fA["pos"]
    eps = episode_list(B); eps2 = eps[:i1] + [e[:e.index(b"=") + 1] + mutate(e, e.index(b"=") + 1, 3 + k)[e.index(b"=") + 1:] if k == 0 else mutate(e, 0, 3 + k)
                                              for k, e in enumerate(eps[i1:])]
    la, lb = L.EpisodeLearner(MODEL_SEED, eps, cutsA), L.EpisodeLearner(MODEL_SEED, eps2, cutsA)
    la.advance_to(i1); lb.advance_to(i1)
    ra, ga = la.respond_logits(fA["prompts"][0]); rb, gb = lb.respond_logits(fA["prompts"][0])
    out["A_first"] = dict(probe_at_episode=i1, episodes_changed=sum(x != y for x, y in zip(eps, eps2)), updates=la.updates,
                          response=[ra.decode("latin-1"), rb.decode("latin-1")], logits_bitwise_equal=bool(T.equal(ga, gb)))
    out["passed"] = (out["B_first"]["logits_bitwise_equal"] and out["B_first"]["batched_responses_equal"] and out["B_panelA"]["responses_equal"]
                     and out["B_panelA"]["logits_bitwise_equal_first_40"] and out["A_first"]["logits_bitwise_equal"]
                     and out["B_first"]["bytes_changed"] > 0 and out["B_panelA"]["bytes_changed"] > 0 and out["A_first"]["episodes_changed"] > 0)
    return out, keepB, panelA


# ---------------------------------------------------------------- F2
@T.no_grad()
def broken_first_response(learner):
    """The response a probe would give after a chunk whose last INPUT is '=' (its last target the first answer digit): the first digit read
    from the carried state with the updated weights, then two more bytes fed back."""
    m = learner.m; a = learner.cursor - 1
    x = learner.xs[a:a + 1][None]                     # the '=' just read
    z1 = T.tanh(m.l1(m.book(x)))
    lg = m.l3(T.tanh(m.l2(z1) + m.lm(learner.h.transpose(0, 1))))[0, -1]
    y = int(lg.argmax())
    rest, _ = L.generate(m, learner.h, [[y]], n=2)
    return bytes([y]) + rest[0], lg


def f2(cfg, B, taught):
    s = B["streams"]["ordinary"]
    ev, cuts, end = SC.plan(B, "ordinary", "B", taught)
    first = next(e for e in ev if e["type"] == "first"); q1 = first["pos"]          # first answer byte at q1
    good_cuts = cuts
    assert q1 % L.CHUNK and (q1 + 1) % L.CHUNK, "the nominal grid already falls at the probe"
    bad_cuts = [c for c in cuts if c != q1] + [q1 + 1]                             # the chunk ends one target later: "=" as last input
    out = {}
    for name, cc in (("correct", good_cuts), ("broken", bad_cuts)):
        base = stream_learner(B, s, cc)
        start = max([0] + [g for g in base.grid if g < q1 and g != q1]) if name == "broken" else None
        res = []
        if name == "correct":
            base.advance_to(q1)
        else:
            base.advance_to(start)
        for d in range(10):
            lr = copy.deepcopy(base)
            ys = lr.ys.clone(); ys[q1] = 48 + d; lr.ys = ys       # the revealed first digit set to d (target position q1)
            if name == "correct":
                r, lg = lr.respond_logits(b"")
            else:
                lr.advance_to(q1 + 1)
                r, lg = broken_first_response(lr)
            res.append(dict(revealed=d, response=r.decode("latin-1"), first_byte_is_revealed=r[:1] == bytes([48 + d]), logits=lg))
        ref = res[0]["logits"]
        out[name] = dict(responses=[x["response"] for x in res], follows_reveal=sum(x["first_byte_is_revealed"] for x in res),
                         distinct_responses=len({x["response"] for x in res}),
                         logits_identical_across_reveals=all(T.equal(ref, x["logits"]) for x in res),
                         max_logit_change=float(max((x["logits"] - ref).abs().max() for x in res)))
    # arm A: the batch that holds the probe episode trained before it is answered
    evA, cutsA, endA = SC.plan(B, "ordinary", "A", taught)
    fA = next(e for e in evA if e["type"] == "first"); i1 = fA["pos"]
    eps = episode_list(B)
    for name, cc in (("A_correct", cutsA), ("A_broken", [c for c in cutsA if c != i1])):
        base = L.EpisodeLearner(MODEL_SEED, eps, cc)
        start = max(g for g in [0] + base.grid if g <= i1)
        nxt = min(g for g in base.grid if g > i1) if name == "A_broken" else i1
        base.advance_to(start if name == "A_broken" else i1)
        res = []
        for d in range(10):
            lr = copy.deepcopy(base)
            e = bytearray(eps[i1]); q = e.index(b"="); e[q + 1] = 48 + d
            lr.eps = eps[:i1] + [bytes(e)] + eps[i1 + 1:]
            if name == "A_broken":
                lr.advance_to(nxt)
            r, lg = lr.respond_logits(fA["prompts"][0])
            res.append(dict(revealed=d, response=r.decode("latin-1"), first_byte_is_revealed=r[:1] == bytes([48 + d]), logits=lg))
        ref = res[0]["logits"]
        out[name] = dict(responses=[x["response"] for x in res], follows_reveal=sum(x["first_byte_is_revealed"] for x in res),
                         distinct_responses=len({x["response"] for x in res}),
                         logits_identical_across_reveals=all(T.equal(ref, x["logits"]) for x in res),
                         max_logit_change=float(max((x["logits"] - ref).abs().max() for x in res)))
    # first attempt (run 1 of stage 1, kept as it ran): at this single early probe the broken boundary moved the logits but not the bytes
    out["first_probe_only_criterion_as_first_written"] = dict(
        condition="correct invariant AND broken logits move AND broken response bytes differ across the ten reveals (both arms' logits)",
        held=bool(out["correct"]["logits_identical_across_reveals"] and not out["broken"]["logits_identical_across_reveals"]
                  and out["broken"]["distinct_responses"] > 1))
    out["B_all_acquire"] = broken_all(B, taught, "B")
    out["A_all_acquire"] = broken_all(B, taught, "A")
    out["passed"] = (out["correct"]["logits_identical_across_reveals"] and not out["broken"]["logits_identical_across_reveals"]
                     and out["A_correct"]["logits_identical_across_reveals"] and not out["A_broken"]["logits_identical_across_reveals"]
                     and all(out[k]["probes_whose_response_bytes_moved_with_the_reveal"] > 0 and out[k]["mean_pull_toward_revealed_digit"] > 0
                             for k in ("B_all_acquire", "A_all_acquire")))
    out["note"] = ("'broken' cuts the chunk one target later than the rule, so that its last input is '=' and its last target the first answer "
                   "digit; the condition holds when the correct boundary's response does not move with the reveal and the broken one's does")
    return out


def broken_all(B, taught, arm):
    """Every acquisition-phase first-encounter probe with the broken boundary (the update before it has used its first answer digit as a
    target): the revealed first digit set to 0..9 in turn. Per probe: do the response bytes differ across the ten reveals, how many of the
    ten responses start with the revealed digit (1 if the response does not move), and the pull: log p(d | d revealed) minus the mean over
    the other nine reveals of log p(d | other revealed), averaged over d (0 when nothing leaks)."""
    ev, cuts, end = SC.plan(B, "ordinary", arm, taught)
    acq = [e for e in ev if e["type"] == "first" and e["probe"]["set"] == "acquire"]
    if arm == "B":
        pos = [e["pos"] for e in acq]
        assert all(q % L.CHUNK and (q + 1) % L.CHUNK for q in pos)
        base = stream_learner(B, B["streams"]["ordinary"], [c for c in cuts if c not in pos] + [q + 1 for q in pos])
    else:
        base = L.EpisodeLearner(MODEL_SEED, episode_list(B), [c for c in cuts if c not in {e["pos"] for e in acq}])
    moved = follows = 0; pulls = []
    for e in acq:
        q = e["pos"]
        start = max([0] + [g for g in base.grid if g <= q - (0 if arm == "A" else 1)])
        if arm == "B":
            start = max([0] + [g for g in base.grid if g < q + 1 and g != q + 1])
        base.advance_to(start)
        resp, lps = [], []
        for d in range(10):
            lr = copy.deepcopy(base)
            if arm == "B":
                ys = lr.ys.clone(); ys[q] = 48 + d; lr.ys = ys
                lr.advance_to(q + 1)
                r, lg = broken_first_response(lr)
            else:
                ep = bytearray(lr.eps[q]); k = ep.index(b"="); ep[k + 1] = 48 + d
                lr.eps = lr.eps[:q] + [bytes(ep)] + lr.eps[q + 1:]
                lr.advance_to(min(g for g in lr.grid if g > q))
                r, lg = lr.respond_logits(e["prompts"][0])
            resp.append(r); lps.append(T.log_softmax(lg.double(), -1)[48:58])
        moved += len(set(resp)) > 1
        follows += sum(r[:1] == bytes([48 + d]) for d, r in enumerate(resp))
        pulls.append(float(sum(lps[d][d] - sum(lps[o][d] for o in range(10) if o != d) / 9 for d in range(10)) / 10))
    return dict(probes=len(acq), probes_whose_response_bytes_moved_with_the_reveal=moved, responses_starting_with_revealed_digit=follows,
                of=10 * len(acq), mean_pull_toward_revealed_digit=sum(pulls) / len(pulls), min_pull=min(pulls), max_pull=max(pulls),
                updates_at_last_probe=base.updates)


# ---------------------------------------------------------------- F3
def f3(learner, panel):
    """learner: arm B, standing at panel A's boundary."""
    before = learner.snapshot()
    tw = [i for i, m in enumerate(panel["meta"]) if m["kind"] == "twin"]
    pairs = [(panel["prompts"][i], panel["prompts"][i + 1]) for i in tw[::2]]
    alone, after_first, broken = [], [], []
    for m1, m2 in pairs[:24]:
        r2, g2 = learner.respond_logits(m2)
        learner.respond_logits(m1)
        r2b, g2b = learner.respond_logits(m2)
        alone.append(r2); after_first.append(T.equal(g2, g2b) and r2 == r2b)
        # broken protocol: the second member read from the state the first member and its answer left behind
        with T.no_grad():
            h = learner.boundary_state()
            res, _ = L.generate(learner.m, h, [[learner.pending_input()] + list(m1)])
            x = T.tensor([[learner.pending_input()] + list(m1) + list(res[0])])
            _, h1 = learner.m(x, h)
            r3, g3 = L.generate(learner.m, h1, [list(m2)])
        broken.append(not T.equal(g2, g3[0]))
    batched = learner.respond(panel["prompts"])
    single = [learner.respond_logits(p)[0] for p in panel["prompts"]]
    after = learner.snapshot()
    out = dict(twin_pairs_tested=len(after_first), second_member_unaffected=sum(after_first),
               broken_protocol_changed_second_member_logits=sum(broken),
               batched_vs_one_by_one_prompts=len(single), batched_vs_one_by_one_mismatches=sum(a != b for a, b in zip(batched, single)),
               state_unchanged_by_panel=same_tensors(before, after))
    out["passed"] = (out["second_member_unaffected"] == out["twin_pairs_tested"] and out["broken_protocol_changed_second_member_logits"] > 0
                     and out["batched_vs_one_by_one_mismatches"] == 0 and out["state_unchanged_by_panel"])
    return out


# ---------------------------------------------------------------- F4
class LeakyStreamLearner(L.StreamLearner):
    """Broken on purpose: after answering, keeps the state its first prompt and answer reached."""

    def respond(self, prompts):
        out = super().respond(prompts)
        with T.no_grad():
            x = T.tensor([[self.pending_input()] + list(prompts[0]) + list(out[0])])
            _, self.h = self.m(x, self.h)
        return out


def f4(cfg, B, taught):
    out = {}
    s = B["streams"]["ordinary"]
    ev, cuts, end = SC.plan(B, "ordinary", "B", taught)
    pA = next(e for e in ev if e["type"] == "panel" and e["name"] == "A")["pos"]
    upto = [e for e in ev if e["pos"] <= pA]
    snaps = {}
    for name, cls, on in (("on", L.StreamLearner, True), ("off", L.StreamLearner, False), ("leaky", LeakyStreamLearner, True)):
        lr = cls(MODEL_SEED, s, cuts)
        for e in upto:
            lr.advance_to(e["pos"])
            if on:
                lr.respond(e["prompts"])
        stop = lr.grid[lr.grid.index(pA) + 10]           # ten chunks past panel A
        lr.advance_to(stop)
        snaps[name] = (lr.snapshot(), lr.loss_log, lr.cuts_used, lr.gi, lr.responses)
    on, off, lk = snaps["on"], snaps["off"], snaps["leaky"]
    out["B"] = dict(events=len(upto), probes_answered=on[4], updates=int(on[0]["counters"][1]), stop=stop,
                    on_equals_off=same_tensors(on[0], off[0]) and on[1] == off[1] and on[2] == off[2] and on[3] == off[3],
                    leaky_differs=not same_tensors(on[0], lk[0]), leaky_differing_entries=len(diff_keys(on[0], lk[0])))
    evA, cutsA, endA = SC.plan(B, "ordinary", "A", taught)
    iA = next(e for e in evA if e["type"] == "panel" and e["name"] == "A")["pos"]
    uptoA = [e for e in evA if e["pos"] <= iA]
    eps = episode_list(B)
    snaps = {}
    for name, on_ in (("on", True), ("off", False)):
        lr = L.EpisodeLearner(MODEL_SEED, eps, cutsA)
        for e in uptoA:
            lr.advance_to(e["pos"])
            if on_:
                lr.respond(e["prompts"])
        lr.advance_to(lr.grid[lr.grid.index(iA) + 10])   # ten batches past panel A
        snaps[name] = (lr.snapshot(), lr.loss_log, lr.cuts_used, lr.gi, lr.responses)
    on, off = snaps["on"], snaps["off"]
    out["A"] = dict(events=len(uptoA), probes_answered=on[4], updates=int(on[0]["counters"][1]),
                    on_equals_off=same_tensors(on[0], off[0]) and on[1] == off[1] and on[2] == off[2] and on[3] == off[3])
    out["compared"] = "every weight, every AdamW state tensor (step, exp_avg, exp_avg_sq), the carried state, cursor, updates, targets, inputs read, cuts used, schedule index, the loss log"
    out["passed"] = out["B"]["on_equals_off"] and out["B"]["leaky_differs"] and out["A"]["on_equals_off"]
    return out


def run():
    T.set_num_threads(1)
    t0 = time.process_time()
    cfg, B, taught = setup()
    r1, learnerB, panelA = f1(cfg, B, taught)
    r3 = f3(learnerB, panelA)
    r2 = f2(cfg, B, taught)
    r4 = f4(cfg, B, taught)
    out = dict(world=WORLD, model_seed=MODEL_SEED, configuration=cfg, F1_reveal_invariance=r1, F2_boundary_rule_can_fail=r2, F3_twins=r3,
               F4_probe_on_off_parity=r4, cpu_seconds=round(time.process_time() - t0, 1))
    out["passed"] = all(x["passed"] for x in (r1, r2, r3, r4))
    return out


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps(r, indent=1, default=str)[:6000])
