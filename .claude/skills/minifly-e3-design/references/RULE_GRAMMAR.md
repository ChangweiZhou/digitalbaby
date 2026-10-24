# The declarative decision rule: primary, gates and guards

The rule is data in ENDPOINT_LOCK.json. The program's `decide` (tools/minifly_check/decide.py) evaluates it on cells
computed by the program's scorer; no delivery code computes the gated metric or the verdict (DEC-01). Normative text:
`tools/minifly_check/SPEC.md` sections 7.4 and 9.1-9.5. Field types and enums: `tools/minifly_check/SCHEMAS.md`, which
wins wherever it is more specific than the examples below (in particular for the field names inside gates and guards).

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`.

## 1. primary: field meanings

| Field | Meaning |
|---|---|
| task_id | The task.id this rule scores |
| metric | exact_own_eos or balanced_exact_own_eos. Anything else forces gate_type component (LOCK-03) |
| unconditional | true: every item counts, never a subset selected by the arm's own behaviour (LOCK-03) |
| pool | TAUGHT, OLD, NEW, HELD or PRESENTATION. E3 uses HELD |
| checkpoint | Which of checkpoints {prefix, old_end, final} is scored; the objective uses final |
| strata_counted | Strata whose determined items count. E3: only S2, S3, S4 (STR-01) |
| candidate_arm | The arm id the claim is about |
| comparator_arms | Arm ids whose exact rates enter v_comp_w (include the causal control; for E3, no_prior_teaching). Every declared arm of role no_prior_teaching, shuf, frozen or lesion enters v_comp_w as well, listed or not (section 3) |
| ladder | auto: the program shortcut ladder for the pool (section 4). Never a hand-picked list |
| chance | Chance level for the metric on this pool |
| margin_abs | Required headroom over the comparator; at least CONSTRAINTS margin_abs (0.05 draft) |
| absolute_floor | mean(v_cand) must reach it; at least chance + absolute_floor_above_chance (0.05 draft) |
| per_world_floor | min over worlds of d_w must reach it; at least CONSTRAINTS per_world_floor (-0.05 draft) |
| aggregator | {type: paired_lcb, alpha} or {type: k_of_n, k} or {type: all_worlds} |
| min_worlds | At least min_worlds_nonpilot (8 draft) unless the run is a PILOT |
| n_items_per_pool | Denominators per pool. Missing rows count as missing, never as a smaller denominator (LAD-01) |
| rule_type | declarative. custom only under DEC-01 (section 6) |

`gates[]` are the delivery's plumbing gates: named pass/fail conditions on its wiring, for example that the oracle arm's
answer reaches the native output path, or that a port opens only under its declared condition. Each gate carries what it
measures (arm, pool, checkpoint or fixture params), a comparison and a threshold. decide requires every gate to pass,
and ADM-GATE requires at least one negative fixture (result false) and one positive fixture (result true) per gate in
FIXTURES.jsonl: a gate that cannot fail is not a gate (v5.1's port gate).

`guards[]` protect what the primary does not measure. Whenever an OLD pool exists, both of these are required (LOCK-04):
- old_drop: mean over worlds of (OLD_end - OLD_final) <= max_drop. Under paired_lcb the upper confidence bound is used.
  max_drop is at most CONSTRAINTS old_max_drop (0.10 draft).
- old_acq: mean over worlds of (OLD_end - best exposure-independent shortcut on OLD) >= margin. margin is at least
  CONSTRAINTS old_acq_margin (0.05 draft). This is the acquisition floor: zero drop from shortcut level still fails.

Every guard needs its own registered prediction (`target: guard:<id>`; LOCK-11).

## 2. How decide evaluates the rule

Preconditions, in order:
1. INERT alarm (INERT-01): INVALID, reason INTEGRITY_ALARM.
2. Testbed failure (ADM-TB-CEIL or ADM-TB-FLOOR): CANNOT_TEST_CLAIM_AS_SPECIFIED.
3. Control failure (ADM-CTRL-ZERO, ADM-CTRL-ISO or ADM-CTRL-DOSE): INVALID, reason TREATMENT_NOT_DELIVERED.
4. Any missing cell: INCOMPLETE, with a shadow verdict computed on the complete paired worlds.
5. Non-finite values: INVALID, reason NONFINITE.

Then, per world w, on the counted items (determined, in strata_counted, answer_path native):
- v_cand = exact rate of the candidate;
- v_comp_w = max(chance, each decision-comparator arm's exact rate, the best ladder member for the pool), where the
  decision comparator is comparator_arms plus every declared no_prior_teaching, shuf, frozen and lesion arm;
- d_w = v_cand - v_comp_w.

Aggregators:
- paired_lcb: LCB = mean(d) - t(1 - alpha, n - 1) * sd / sqrt(n). If sd == 0, the Hoeffding bound
  LCB = mean(d) - 2 * sqrt(ln(1/alpha) / (2n)) is used. The primary passes when LCB > margin_abs.
- k_of_n: count(d_w > margin_abs) >= k.
- all_worlds: every d_w > margin_abs.

The rule also requires mean(v_cand) >= absolute_floor, min(d_w) >= per_world_floor, every gate passing and every guard
passing. If any world has role technical, dev or rehearsal, the verdict is DEV_NOT_EVIDENCE with a shadow_verdict.
Otherwise it is PASS when everything passes, else FAIL. The verdict enum is closed: PASS, FAIL, INVALID, INCOMPLETE,
DEV_NOT_EVIDENCE, CANNOT_TEST_CLAIM_AS_SPECIFIED. The alpha used must not exceed the CONFIRM_LEDGER allocation (CONF-01).

Worked example (8 official MAIN worlds, paired_lcb, alpha 0.05, margin_abs 0.05):

| | w1 | w2 | w3 | w4 | w5 | w6 | w7 | w8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v_cand | 0.62 | 0.55 | 0.70 | 0.48 | 0.66 | 0.59 | 0.51 | 0.64 |
| v_comp_w | 0.31 | 0.25 | 0.34 | 0.29 | 0.33 | 0.27 | 0.31 | 0.30 |
| d_w | 0.31 | 0.30 | 0.36 | 0.19 | 0.33 | 0.32 | 0.20 | 0.34 |

mean(d) = 0.294, sd = 0.0637, t(0.95, 7) = 1.895, so LCB = 0.294 - 1.895 * 0.0637 / sqrt(8) = 0.251 > 0.05.
mean(v_cand) = 0.594 >= 0.05, and min(d_w) = 0.19 >= -0.05. With OLD drops of 0.04, 0.02, 0.05, 0.02, 0.01, 0.03, 0.04,
0.02 the mean drop is 0.029 with upper bound 0.038 <= 0.10; with OLD_end about 0.50 above the best exposure-independent
shortcut, old_acq passes. If the gates pass, the verdict is PASS. Had w1 been a dev world, it would be
DEV_NOT_EVIDENCE with shadow_verdict PASS. Had all eight d_w been equal, sd == 0 would switch to the Hoeffding bound
(mean - 0.865 at n = 8), a far harder bar. Precondition 1 comes first: INERT-01 raises INTEGRITY_ALARM when the primary
contrast has zero between-world variance with identical values (the [0,0] interval).

## 3. Comparators

- The comparator is never only the arms you chose. chance and the full ladder for the pool are always in the max.
- The decision comparator is comparator_arms plus every declared arm whose role is no_prior_teaching, shuf, frozen or
  lesion (SPEC 9.5 as amended in 20.6). Leaving a control out of comparator_arms never weakens the rule: in verifier
  probe 2c a FROZEN arm equal to the candidate on HELD gave PASS because FROZEN was not declared a comparator; now
  d_w = 0 there and the verdict is FAIL (corpus S59). oracle_plumbing and positive_control arms (meant to be strong)
  and evaluator-only CEILING_*/FLOOR_* arms are never added; the cycle-0 testbed view adds none.
- Every decision-comparator arm needs its primary cells in every world: a missing one is INCOMPLETE (RED-03 drops each).
  VERDICT.primary lists comparator_arms (the decision comparator) and implied_comparator_arms.
- comparator_arms should still name the causal control (no_prior_teaching for E3; CLM-03 reads the declared list) and SHUF.
- INERT-01's zero-variance contrast and LAD-03 read the declared comparator_arms only: a failing candidate at the level
  of a FROZEN or SHUF arm is an honest FAIL through the decision comparator, not an integrity alarm. Any decision
  comparator at or above the candidate is named in the headline suffix `| <arm> >= candidate`.
- LAD-03 (WARN) notes a declared comparator arm below the best shortcut on dev as WEAK_COMPARATOR in the headline.
- LAD-02: decide must not PASS when the candidate's rows are replaced by the best shortcut's predictions.
- STOP-ONLY was judged against a DIRECT comparator that scored zero, while its own stream truncated at one byte scored
  10.645% against STOP's 11.263% on NEW (`minifly_address_capacity_v1_dev_20261023/REPORT.md:7`). Under decide,
  v_comp_w would have been at least FIXED_1's value.

## 4. The shortcut ladder and pool subsets

Members are computed per (world, pool, checkpoint) from the candidate arm's TAUGHT manifest and its own streams:

| Member | Prediction for a scored prompt |
|---|---|
| CONST | The most frequent taught answer; ties go to the lexicographically smallest |
| LENFREQ | The most frequent answer among taught prompts of the same length |
| SUFFIX1..3 | The most frequent answer among taught prompts sharing the last k bytes |
| EXACT | The taught answer for an identical prompt |
| CANON | The taught answer for the same canonical orbit (sorted, normalized operands) |
| NEAREST | The answer of the taught prompt with the most positional byte matches |
| COPY_A, COPY_B | Copy one operand |
| ID_ZERO | If an operand is 0, the other operand |
| DIGIT_NO_CARRY | Digitwise sum mod 10 |
| FIXED_0..8 | The arm's own stream_hex truncated to k bytes, with own EOS assumed at k |
| MODE_REPEAT | The arm's modal output byte repeated to its modal emitted length |

The fallback for every member is CONST. Operand members apply to addition only; skips are recorded.

| Pool | Ladder members used |
|---|---|
| TAUGHT, OLD, NEW | Exposure-independent only: CONST, LENFREQ, SUFFIXk, COPY_A/B, ID_ZERO, DIGIT_NO_CARRY, FIXED_k, MODE_REPEAT |
| PRESENTATION | The full ladder except CANON |
| HELD | The full ladder |

DIGIT_NO_CARRY is exact on every no-carry item, so it sets a high bar on S2. On counted held items EXACT and CANON have
no taught answer (the canonical orbit is untaught by definition) and fall back to CONST; they bound S1, not S2+.
FIXED_k asks whether the arm's own bytes with a fixed stopping length would do as well, which is the STOP-ONLY case.
MODE_REPEAT catches an arm that emits the same bytes regardless of the prompt. LAD-04 (WARN) flags a modal output
fraction above 0.9 or an entropy below 0.5 bit in any pool.

## 5. Examples

Channel names (`unit_store` and so on) stand for the learner's own receipt channels. Only rule-relevant fields are shown.

E1, taught retention (no held claim; scored items are taught, so their canonical orbit is taught and the checker places
them in S1 unless trivial):

```json
{
  "claim_level_target": "E1",
  "intent": "diagnostic",
  "arms": [
    {"id": "W", "role": "candidate", "zero_channels": []},
    {"id": "SHUF", "role": "shuf", "zero_channels": []},
    {"id": "NOWRITE", "role": "lesion", "zero_channels": ["unit_store"], "min_realized_dose": 0.05},
    {"id": "POS", "role": "positive_control", "zero_channels": []}
  ],
  "causal_control_arm": "NOWRITE",
  "positive_control_arm": "POS",
  "primary": {
    "task_id": "add1", "metric": "exact_own_eos", "unconditional": true,
    "pool": "TAUGHT", "checkpoint": "final", "strata_counted": ["S1"],
    "candidate_arm": "W", "comparator_arms": ["NOWRITE", "SHUF"], "ladder": "auto",
    "chance": 0.0, "margin_abs": 0.05, "absolute_floor": 0.05, "per_world_floor": -0.05,
    "aggregator": {"type": "k_of_n", "k": 7}, "min_worlds": 8,
    "n_items_per_pool": {"TAUGHT": 32, "OLD": 32}, "rule_type": "declarative"
  },
  "gates": [],
  "guards": [{"id": "old_drop", "kind": "old_drop", "pool": "OLD", "max_drop": 0.10},
             {"id": "old_acq", "kind": "old_acq", "pool": "OLD", "margin": 0.05}]
}
```

An E1 PASS earns E1 only: an exact-key table explains taught answers. It opens no held E3 (section 10 of the skill).

E3, held composition from column units (diagnostic):

```json
{
  "claim_level_target": "E3",
  "intent": "diagnostic",
  "units": {"witness": "column_closure", "commutative": true},
  "arms": [
    {"id": "W", "role": "candidate", "zero_channels": []},
    {"id": "NPT", "role": "no_prior_teaching", "zero_channels": ["unit_store"], "teaching_withheld": ["teach_units"]},
    {"id": "SHUF", "role": "shuf", "zero_channels": [], "declared_changed": ["unit_store"]},
    {"id": "ORACLE", "role": "oracle_plumbing", "zero_channels": [], "declared_changed": ["readout"]}
  ],
  "causal_control_arm": "NPT",
  "positive_control_arm": "ORACLE",
  "primary": {
    "task_id": "add2", "metric": "exact_own_eos", "unconditional": true,
    "pool": "HELD", "checkpoint": "final", "strata_counted": ["S2", "S3", "S4"],
    "candidate_arm": "W", "comparator_arms": ["NPT", "SHUF"], "ladder": "auto",
    "chance": 0.0, "margin_abs": 0.05, "absolute_floor": 0.05, "per_world_floor": -0.05,
    "aggregator": {"type": "paired_lcb", "alpha": 0.05}, "min_worlds": 8,
    "n_items_per_pool": {"TAUGHT": 32, "HELD": 48, "OLD": 32, "NEW": 16}, "rule_type": "declarative"
  },
  "gates": [{"id": "oracle_answer_path", "arm": "ORACLE", "pool": "HELD", "checkpoint": "final", "op": ">=", "value": 0.9}],
  "guards": [{"id": "old_drop", "kind": "old_drop", "pool": "OLD", "max_drop": 0.10},
             {"id": "old_acq", "kind": "old_acq", "pool": "OLD", "margin": 0.05}],
  "udi": null,
  "worlds": {"official": {"mode": "listed", "ids": ["<8 fresh world ids>"], "n": 8},
             "official_role_intended": "MAIN"}
}
```

A PASS here earns E3 (CLM-03 also needs ADM-EXPO PASS and headroom over the best shortcut of at least margin_abs) with
reuse_specificity not_tested, on MAIN worlds, with no confirm wording.

E3 promotion (SEALED_CONFIRM, with UDI): the E3 lock above, changed in these fields only:

```json
{
  "intent": "promotion",
  "udi": {"variants_manifest": "manifests/UDI_VARIANTS.json", "alpha": 0.05},
  "worlds": {"official": {"mode": "sealed", "ids": [], "n": 8}, "official_role_intended": "SEALED_CONFIRM"}
}
```

- Official world ids are derived after the owner's GO from the seal (SEAL-04); the UDI dependent sets use the seal
  purpose 'udi'.
- primary.aggregator.alpha must not exceed the alpha_allocated that `seal reveal` records for this family attempt
  (CONF-01; default schedule 0.05 * 2^-j). The lock is frozen first, so set alpha from the schedule for the family's
  next attempt.
- `MFC oc <dir>` must show P(pass | target) >= 0.8 (OC-01). Promotion additionally needs UDI LCB > 0 (UDI-01, planned in
  checker 1.0.0; CLM-03),
  REVIEW and replay (CLM-09, REV-08).

## 6. Why no delivery decision code is allowed

A delivery's own reducer is written by the party whose result it judges, can see which inputs are fixtures, and has
repeatedly failed open:
- LEARNED-BANK `summarize.py:92-100` (origin/charming-head `minifly_learned_bank_v1_20261024/summarize.py`):
  `v["gate_oracle"] is not False` passes a gate whose value is missing, the overall verdict is
  `n_ok == len(verdict)` over whatever worlds happen to be present, and a 'partial' grade sits outside any verdict
  enum (proposed in `cycles/CYCLE2.md:101-110`).
- V6-R1 `minifly_v6_r1_dev_20261023/audit_results.py:97` writes `exposure_overlap=0` and
  `all_rows_decoded_from_native_margins=True` as literals into a PASS audit (LOCK-08 now fails this).

So the verdict comes only from the program's decide (DEC-01: decide_version, both lock shas and the cells sha in
VERDICT.json), which is corpus-tested and mutation-tested and knows nothing about which delivery it is judging.

`rule_type: custom` exists for rules the grammar cannot express. It needs a UD, a REVIEW whose author is not the
executor, and 50 reviewer-seeded property fixtures that pass (DEC-01). It runs only in a subprocess (`python3 -I`, 20 s
timeout, synthetic or canonical cells only), and a return outside the verdict enum is FAIL.

## 7. Rule qualification at the roster gate (RED-01..11)

The checker builds synthetic cells from your own arms, worlds, pools and strata and runs your rule:

| Check | Synthetic case | Required outcome |
|---|---|---|
| RED-01 | Clean positive: candidate 0.9, all else at chance | PASS (the rule is reachable) |
| RED-02 | Candidate equal to each decision-comparator arm (declared and implied), to CONST/FIXED_1/MODE_REPEAT and to chance | not PASS |
| RED-03 | A decision-comparator arm missing (each, declared and implied), or the ladder missing | INCOMPLETE |
| RED-04 | Fewer than min_worlds | not PASS |
| RED-05 | None or NaN | INVALID |
| RED-06 | All-dev worlds | DEV_NOT_EVIDENCE |
| RED-07 | Candidate at comparator + margin - 1e-9 | not PASS |
| RED-08 | Gain only on S0/S1 | not PASS |
| RED-09 | OLD guard failure | not PASS |
| RED-10 | The shortcut substituted for the candidate | not PASS |
| RED-11 | OLD_END at shortcut level with zero drop | not PASS |

A RED-01 failure means the rule cannot pass even a clean positive (for example margin 0.6 with 8 items per world:
corpus fixture C05c). Amend it before any learner-arm outcome; no UD is needed while none exists.
