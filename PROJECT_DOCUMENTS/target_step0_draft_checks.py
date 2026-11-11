"""Coordinator's checks of the two corrections Codex made to the step 0 draft (PROJECT_DOCUMENTS/MINIFLY_PROGRESS_SURVEY_AND_TARGET_ASSAY_20261110.md),
and of its question about the prior-teaching control. Standard library only; no learner. Writes TARGET_STEP0_DRAFT_CHECKS.json beside this file.
The world follows OPUS_TASK_TARGET_TEST_STEP0_20261111.md: 24 contexts, two families of 12 cues, outcomes (v + u) mod 97 and (v - u') mod 97."""
import json, random, pathlib, itertools
M, NCTX, NCUE, FILL = 97, 24, 12, b"!#%&*+,-"
out = {}

# ---- correction 1: the draft's counts
out["1_counts_in_the_draft"] = dict(pairs_per_family=NCTX * NCUE, taught=216, held_and_eventually_revealed=72, left_for_sealed_twins=NCTX * NCUE - 216 - 72,
                                    twins_asked_per_phase=150)

def world(seed, shared_context_values=True):
    r = random.Random(seed)
    v1 = [r.randrange(M) for _ in range(NCTX)]
    v2 = v1 if shared_context_values else [r.randrange(M) for _ in range(NCTX)]
    u1 = [r.randrange(M) for _ in range(NCUE)]; u2 = [r.randrange(M) for _ in range(NCUE)]
    return v1, v2, u1, u2
def outcome(w, fam, x, c):
    v1, v2, u1, u2 = w
    return (v1[x] + u1[c]) % M if fam == 0 else (v2[x] - u2[c]) % M
def episode(fam, x, c, filler):
    return bytes([65 + x]) + filler + bytes([97 + c + 12 * fam]) + b"="
def split(seed, n_first=48, n_sealed=24):
    """Codex's illustrative split per family: 216 taught, 48 first-encounter, 24 sealed for good. Sealed pairs come as twins: one cue, two contexts."""
    r = random.Random(seed + 1); S = {}
    for fam in (0, 1):
        while True:
            pairs = [(x, c) for x in range(NCTX) for c in range(NCUE)]; sealed = []
            for c in r.sample(range(NCUE), n_sealed // 2):
                x1, x2 = r.sample(range(NCTX), 2); sealed += [(x1, c), (x2, c)]
            rest = [p for p in pairs if p not in sealed]; r.shuffle(rest)
            S[fam] = dict(sealed=sealed, first=rest[:n_first], taught=rest[n_first:])
            if len(S[fam]["taught"]) == 216: break
    return S

# ---- linear algebra over Z_97: is a held answer fixed by the taught outcomes?
def rowspace(rows):
    rows = [r[:] for r in rows]; piv = []
    for col in range(len(rows[0])):
        k = next((i for i in range(len(piv), len(rows)) if rows[i][col] % M), None)
        if k is None: continue
        rows[len(piv)], rows[k] = rows[k], rows[len(piv)]
        inv = pow(rows[len(piv)][col], M - 2, M); rows[len(piv)] = [a * inv % M for a in rows[len(piv)]]
        for i in range(len(rows)):
            if i != len(piv) and rows[i][col] % M:
                f = rows[i][col]; rows[i] = [(a - f * b) % M for a, b in zip(rows[i], rows[len(piv)])]
        piv.append(col)
    return rows[:len(piv)], piv
def determined(basis, piv, target):
    t = target[:]
    for row, col in zip(basis, piv):
        if t[col] % M:
            f = t[col]; t = [(a - f * b) % M for a, b in zip(t, row)]
    return not any(a % M for a in t)
def eq(fam, x, c, shared):
    """unknowns: v1 (24), v2 (24, the same columns as v1 when shared), u1 (12), u2 (12)"""
    row = [0] * (2 * NCTX + 2 * NCUE)
    row[x if (fam == 0 or shared) else NCTX + x] = 1
    row[2 * NCTX + c + NCUE * fam] = 1 if fam == 0 else M - 1
    return row

res = {}
for seed in (1, 2, 3):
    w = world(seed); S = split(seed); r = random.Random(seed + 2)
    # the stream of taught episodes, 40 renderings of each taught pair, filler 0-14 uniform
    table10, table40, exact = {}, {}, {}
    for fam in (0, 1):
        for (x, c) in S[fam]["taught"]:
            for _ in range(40):
                e = episode(fam, x, c, bytes(r.choice(FILL) for _ in range(r.randrange(15)))); o = outcome(w, fam, x, c)
                table10[e[-10:]] = o; table40[e[-40:]] = o; exact[(fam, x, c)] = o
    # sealed twins: 150 renderings per family from the 12 sealed twin pairs, filler of 10 to 14 shared by the two members
    twins = []; dropped = 0
    for fam in (0, 1):
        tw = [(S[fam]["sealed"][i], S[fam]["sealed"][i + 1]) for i in range(0, 24, 2)]
        # the draft's rule (its check 6): twin contexts must have different values. Drawn at random, some coincide; they are dropped and counted.
        same = [q for q in tw if outcome(w, fam, q[0][0], q[0][1]) == outcome(w, fam, q[1][0], q[1][1])]; dropped += len(same)
        tw = [q for q in tw if q not in same]
        for i in range(150):
            (x1, c), (x2, _) = tw[i % len(tw)]; f = bytes(r.choice(FILL) for _ in range(r.randrange(10, 15)))
            twins.append((fam, x1, x2, c, f))
    def both(resp):
        return sum(resp(fam, x1, c, f) == outcome(w, fam, x1, c) and resp(fam, x2, c, f) == outcome(w, fam, x2, c) for fam, x1, x2, c, f in twins)
    def either(resp):
        return sum((resp(fam, x1, c, f) == outcome(w, fam, x1, c)) + (resp(fam, x2, c, f) == outcome(w, fam, x2, c)) for fam, x1, x2, c, f in twins)
    look10 = lambda fam, x, c, f: table10.get(episode(fam, x, c, f)[-10:])
    look40 = lambda fam, x, c, f: table40.get(episode(fam, x, c, f)[-40:])
    lookexact = lambda fam, x, c, f: exact.get((fam, x, c))
    # the rule-aware evaluator: solves the values from the taught outcomes (differences are enough), then answers
    basis, piv = rowspace([eq(fam, x, c, True) + [outcome(w, fam, x, c)] for fam in (0, 1) for (x, c) in S[fam]["taught"]])
    def solve(fam, x, c):
        t = eq(fam, x, c, True) + [0]
        for row, col in zip(basis, piv):
            if t[col] % M:
                fct = t[col]; t = [(a - fct * b) % M for a, b in zip(t, row)]
        return (-t[-1]) % M if not any(a % M for a in t[:-1]) else None
    full = lambda fam, x, c, f: solve(fam, x, c)
    # the same evaluator restricted to the shared suffix: it sees the cue, not the context, so it must give one answer per suffix
    r2 = random.Random(seed + 3)
    suffix_only = lambda fam, x, c, f: solve(fam, random.Random(bytes([fam, c]) + f).randrange(NCTX), c)
    res[seed] = dict(n_twins=len(twins), twin_pairs_dropped_for_equal_values=dropped, distinct_sealed_pairs=sum(len(S[f]["sealed"]) for f in (0, 1)),
                     lookup_10_bytes_both=both(look10), lookup_40_bytes_both=both(look40), lookup_40_bytes_members_right=either(look40),
                     exact_key_table_both=both(lookexact),
                     rule_aware_full_episode_both=both(full), rule_aware_suffix_only_both=both(suffix_only),
                     rule_aware_suffix_only_members_right=either(suffix_only))
out["2_visibility_and_memorisation_on_sealed_twins"] = res

# ---- the prior-teaching control: remove every family-1 teaching episode of one context; is its family-1 answer still fixed?
ctl = {}
for shared in (True, False):
    w = world(1, shared); S = split(1); x0 = S[0]["sealed"][0][0]; c0 = S[0]["sealed"][0][1]
    rows_all = [eq(fam, x, c, shared) for fam in (0, 1) for (x, c) in S[fam]["taught"]]
    rows_ctl = [eq(fam, x, c, shared) for fam in (0, 1) for (x, c) in S[fam]["taught"] if not (fam == 0 and x == x0)]
    rows_ctl_f1only = [eq(0, x, c, shared) for (x, c) in S[0]["taught"] if x != x0]
    t = eq(0, x0, c0, shared)
    ctl["shared context values" if shared else "separate context values per family"] = dict(
        with_all_teaching=determined(*rowspace(rows_all), t),
        control_family1_teaching_of_that_context_removed=determined(*rowspace(rows_ctl), t),
        same_control_ignoring_family2=determined(*rowspace(rows_ctl_f1only), t))
out["3_prior_teaching_control_answer_still_determined"] = ctl
p = pathlib.Path(__file__).with_name("TARGET_STEP0_DRAFT_CHECKS.json"); p.write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
