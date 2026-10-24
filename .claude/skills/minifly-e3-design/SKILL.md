---
name: minifly-e3-design
description: "Use when designing, locking or analysing any MiniFly endpoint that involves generalization: an E2/E3 or unseen-addition primary; choosing held items, strata, countermodels, arms, causal controls or the OLD guard; writing the primary rule in ENDPOINT_LOCK.json; writing admit.py, receipts or OUTPUTS rows for the cycle-0 testbed screen and behavioral admission; claiming reuse, transfer, rule learning or composition from taught units; or arguing that an E0/E1 component result moves the program toward E3. Makes generalization from unit learned knowledge measurable: units fixed before streams, checker-computed strata S0-S4 (only S2+ count), the determinacy witness, the program shortcut ladder, the declarative decision rule, the no-prior-teaching control and, for E3 promotion, the leave-units-out dependency index, G per sample and per compute, OLD protection with an acquisition floor, native-path purity and the E3 cadence rule."
---

> **If you only read this.** Generalization counts only on determined held items in S2-S4, scored by the program checker against max(chance, comparator arms, shortcut ladder).
> Fix the units and the held set in ENDPOINT_LOCK before any stream exists, and write the rule declaratively: no delivery decision code.
> Run the cycle-0 testbed screen before learner code: at least one ceiling must pass and every floor must fail, or the endpoint cannot test the claim.
> Controls are proved from raw receipts (per-channel digests, write counters at the mutation site). Admission can only stop a run, never support a claim.
> E3 promotion needs UDI LCB > 0 and SEALED_CONFIRM; otherwise report E3 with reuse_specificity not_tested, or the level actually earned.

`MFC` means `python3 -I tools/minifly_check/minifly_check.py`, run from the repository root. Proceed only on exit 0.

- Normative spec: `tools/minifly_check/SPEC.md` (sections 7.4, 8 and 9). Field-level schemas: `tools/minifly_check/SCHEMAS.md`.
- The stage sequence and its commands live in `minifly-research-loop`; this skill says what goes into the endpoint.
- References: [references/RULE_GRAMMAR.md](references/RULE_GRAMMAR.md) (primary, gates, guards, comparators,
  aggregators, examples), [references/ADMISSION_AND_RUNNER_CONTRACT.md](references/ADMISSION_AND_RUNNER_CONTRACT.md)
  (what the runner and admit.py emit), [references/STRATA_WITNESS_UDI.md](references/STRATA_WITNESS_UDI.md) (strata,
  witnesses, UDI and G, with worked addition examples).

## 1. Objective metric and G

- The primary metric is exact_own_eos or balanced_exact_own_eos, with unconditional true and output
  own_bytes_own_eos. Only then is gate_type objective; anything else is forced to component (LOCK-03).
- An item is exact when own_eos is true as recomputed from stream_hex, the emitted bytes equal the expected bytes
  recomputed from the prompt, and answer_path is native (MET-01, MET-02). balanced_exact_own_eos is the mean of the
  per-expected-answer exact rates; use it when answer classes are unbalanced. LEARNED-BANK's D3 held set had 18 of 27
  answers `'1'`, so a constant answer could score up to 0.68 plain but exactly 0.50 balanced (origin/charming-head
  `minifly_learned_bank_v1_20261024/cycles/CYCLE2.md:41`).
- Denominators come from lock.primary.n_items_per_pool, never from the rows present.
- G is the objective-aligned generalization score:

```
G = mean_w [ exact(W, HELD, final, counted) - max(exact(no_prior_teaching), best shortcut, chance) ]
```

- Companions: G_per_sample (G per 1,000 taught presentations, from receipts), G_per_cpu_h (G per lineage-wide CPU-h,
  failed attempts included), OLD_drop and OLD_acq.
- Program progress is G on SEALED_CONFIRM worlds only. MAIN-world G is shown separately as 'unsealed'.

Worked example (full table in STRATA_WITNESS_UDI.md section 6). In one world the candidate scores 0.62 on counted held
items, no_prior_teaching 0.08, the best shortcut DIGIT_NO_CARRY 0.31 and chance 0.0, so the world contributes
0.62 - 0.31 = 0.31. In another, no_prior_teaching is the strongest comparator (0.31 against a shortcut at 0.28), so it
sets the bar. The mean over 8 such worlds is G = 0.294, printed as `G 29.4pp`.

## 2. Units and held targets

- Units are the taught facts the claim says are reused: operand pairs for single-token operands, or column triples
  (a_i, b_i, carry_in_i) for multi-digit operands.
- Declare `units.witness` (pair_rectangle or column_closure) and `units.commutative`.
- Fix the held set in the lock (`split.held_set_sha256`, `split_sha256`) before any stream exists. AMD-05
  recomputes both from the manifests at every stage and every commit after the lock; LOCK-06 requires TAUGHT streams
  to follow the lock commit; the scorer and LAD-01 accept only manifest items in HELD/OLD/NEW cells.
- The checker computes strata S0-S4 from item content and the taught set (STR-01) and determinacy under the witness
  (DET-01). OUTPUTS rows carry no stratum field. Counted items are the determined items in the counted strata; E3
  counts only S2..S4.
- Each counted stratum needs at least CONSTRAINTS.min_items_per_stratum distinct items (8, draft). Too few determined
  counted items is a DET-01 FAIL: the testbed cannot test E3.

| Stratum | Definition |
|---|---|
| S0 trivial | An operand is 0, or the answer equals an operand |
| S1 presentation twin | Canonical orbit taught, or fed with its answer in any exposure stream (spec 20.11; excluded from E3) |
| S2 interpolation | Each operand value appears in its position in some taught pair; no carry |
| S3 carry | As S2, with carry |
| S4 extrapolation | An operand value is unseen in its position, or the answer is longer than every taught answer |

- Put enough S3 and S4 items in the held set. DIGIT_NO_CARRY (digitwise sum mod 10) is exact on every no-carry item,
  so a pool that is mostly S2 leaves little headroom over the ladder.
- V6-R1 drew single-digit operands from 0..9 (`minifly_v6_r1_dev_20261023/v6_data.py:16`), so zero-operand held items
  could occur. Gains confined to them are S0 and never count (corpus fixture C06b).

## 3. Arms by claim level (CONSTRAINTS.required_arms_by_level)

| Level | Required arm roles |
|---|---|
| E1 | candidate, shuf |
| E2 | candidate, shuf, positive_control |
| E3 | candidate, no_prior_teaching, shuf, oracle_plumbing |

- Name causal_control_arm and positive_control_arm in the lock. Arms with role lesion or no_prior_teaching have
  non-empty zero_channels (LOCK-02).
- no_prior_teaching: matched sensory input and timing; the teaching episode claimed to be reused is withheld, and
  teaching_withheld declares it. Both branches get no feedback on the scored instance.
- shuf: an orbit-consistent label permutation that agrees with the truth at about chance (within shuf_band,
  [0.35, 0.65] draft), checked from its TAUGHT manifest (ADM-SHUF). Never a derangement on binary labels: that flips
  every label and teaches the anti-rule (`CYCLE2.md:40`).
- oracle_plumbing and positive_control prove the answer path and the task can be solved through it.
- Ceiling and floor learners are evaluator-only, labelled, drawn from the CONSTRAINTS testbed grids, and never on the
  native answer path.

## 4. The decision rule

Grammar and examples: [references/RULE_GRAMMAR.md](references/RULE_GRAMMAR.md).
- Declarative only: primary, gates and guards in the lock, evaluated by the program's decide. No delivery decision
  code; VERDICT.json must come from decide (DEC-01).
- Comparator per world = max(chance, comparator arms, shortcut ladder for the pool). The comparator arms are
  comparator_arms plus every declared no_prior_teaching, shuf, frozen and lesion arm (never oracle_plumbing,
  positive_control or CEILING_/FLOOR_ arms), so an undeclared FROZEN arm equal to the candidate on HELD never lets it
  pass (corpus S59). An arm of role other joins the comparator too, so relabelling a control 'other' changes
  nothing (spec 20.14, corpus S80). Role candidate belongs to primary.candidate_arm alone: LOCK-02 FAILs a second
  'candidate' arm, and decide counts such an arm as a comparator anyway (corpus S87). A strong arm that is not a
  control is positive_control or oracle_plumbing; those roles stay out of the comparator, so never use them for a
  control. The ladder always includes FIXED_k of the candidate's own stream. STOP-ONLY's 225,792-record confirmation measured STOP against a DIRECT comparator that
  scored zero; its own stream truncated at one byte reached 10.645% against STOP's 11.263% on NEW
  (`minifly_address_capacity_v1_dev_20261023/REPORT.md:7`). In decide the bar would have been at least FIXED_1.
- Margins and floors are at least the CONSTRAINTS floors (LOCK-04): margin_abs, the absolute floor above chance,
  per_world_floor, min_worlds (unless PILOT), the admission thresholds, old_max_drop, old_acq_margin and alpha.
- Aggregators: paired_lcb, k_of_n or all_worlds. A per-world floor and min_worlds always apply.
- rule_type custom needs a UD, a REVIEW whose author is not the executor, and 50 reviewer-seeded property fixtures that
  pass (DEC-01). It runs in a subprocess and a non-enum return is FAIL.
- At the roster gate RED-01..11 qualify your rule on synthetic cells built from your own arms, worlds, pools and strata:
  a clean positive must PASS (reachability), and a shortcut, an S0/S1-only gain, an OLD failure or a near-margin
  candidate must not.

## 5. OLD protection

Declare both guards whenever an OLD pool exists (LOCK-04):
- old_drop: mean(OLD_end - OLD_final) <= max_drop, using the upper bound under paired_lcb. max_drop is at most
  CONSTRAINTS old_max_drop (0.10 draft).
- old_acq: mean(OLD_end - best exposure-independent shortcut on OLD) >= margin. margin is at least CONSTRAINTS
  old_acq_margin (0.05 draft).

old_acq blocks 'protect OLD by never learning OLD': an arm whose OLD_END sits at shortcut level has zero drop and must
still FAIL (RED-11; corpus fixture C06c). V6-R1's OLD guard compared LEARN with NO_OLD on OLD
(`minifly_v6_r1_dev_20261023/audit_results.py:91`) with no floor above the exposure-independent ladder.

## 6. Cycle 0 testbed screen

- admit.py writes CEILING_<cfg> and FLOOR_<cfg> rows on the exact split and budget:
  `MFC launch --stage testbed <dir> -- python3 -I admit.py --testbed`, then `MFC score --role admission <dir>`.
- The checker scores them with the delivery's own rule. At least one ceiling must PASS, and grid coverage must include
  CONSTRAINTS.testbed_ceiling_grid (ADM-TB-CEIL). Every floor must FAIL, and so must the best floor over
  CONSTRAINTS.testbed_floor_grid (ADM-TB-FLOOR).
- Otherwise the verdict is CANNOT_TEST_CLAIM_AS_SPECIFIED, an honest exit: stop, or redesign through an amendment with
  reason testbed_invalid.

LEARNED-BANK's audit (`CYCLE2.md:17-36`) is this screen, run after the learner: on T-MOD exact backprop reached 0.00 / 0.11
held (ceiling fails: untestable), while on CARRY and CMP frozen random features already reached 0.84 / 0.89 and
0.77 / 0.73 (floor passes: cannot show feature learning). PAR separated them cleanly: ceiling 1.00 against floor
0.24 / 0.22.

## 7. Runner contract

Detail: [references/ADMISSION_AND_RUNNER_CONTRACT.md](references/ADMISSION_AND_RUNNER_CONTRACT.md).
- OUTPUTS rows carry stream_hex up to gen_cap_bytes, generated ignoring EOS; the checker recomputes emitted bytes and
  own_eos from it.
- TAUGHT manifests are written per arm (`manifests/TAUGHT/<world>/<arm>.jsonl`).
- Receipts carry per-channel digests at birth, post_first and end, plus write counters incremented at the
  state-mutation site, named in write_counter_site.
- `--only-world` and `--only-arm` flags reproduce one history for replay.
- Rows and receipts are written before any assertion runs.
- All counted runs go through `launch`, one launch per phase. Each writes its runtime closure
  (`runlog/RUNTIME_CLOSURE_<attempt_id>.json`, from sys.modules with the contract helper); a repo module it lists that
  SOURCE_LOCK lacks is a closure gap (RUN-03).
- An interrupted official run continues only with `launch --stage official --resume`: missing units only, one
  completed unit replayed bit-identically into `results/official_replay/` (RUN-02, RUN-04).

Package B's receipt auditor derived its expected write ledger from the same `branch_allows` function that gated the
writes, so the dead N_old_rel control (all 288 old-relation writes performed, digests identical to W in 832/832
receipts) passed it (`minifly_b/REPORT.md:12-26`). The checker takes expectations from the lock's zero_channels and
reads the counts and digests from the raw receipts.

## 8. Behavioral admission

ADMISSION.json is assembled by admit.py from raw rows; the checker recomputes what it can and receipts win over
summaries. The checks: ADM-01, ADM-TB-CEIL, ADM-TB-FLOOR, ADM-CTRL-ZERO, ADM-CTRL-ISO, ADM-CTRL-DOSE, ADM-DIR, ADM-FIX,
ADM-GATE, ADM-INPUT, ADM-ACQ, ADM-SHUF, ADM-EXPO, ADM-BUDGET and ADM-REUSE (T1 only).
- Structure meters (shared-operand vs non-shared code similarity, modal output fraction, code-overlap drift) are always
  reported next to distinguishability. ADM-INPUT requires them; they are reported, not gated.
- Admission can only stop a run. It never supports a claim: a hash code passes distinguishability.
- V6-R1 measured a same-prompt/different-tape distance of 0.977 against a different-prompt/same-tape distance of 0.0137,
  a ratio of 0.014 (`minifly_v6_r1_dev_20261023/REVISION.md:40-52`). input_contrast_min is 2.0 (draft), so ADM-INPUT stops
  this at C1. LOCK-12 asks for the analytic SNR at design, where it was already readable.
- ADM-ACQ: candidate exact own-EOS on TAUGHT at the prefix checkpoint must beat the best exposure-independent shortcut
  by taught_acq_margin, or the verdict is NOT_ACQUIRING.
- ADM-EXPO: canonical held orbits (exact prompts for E2) never occur in a feedback-bearing stream, and every TAUGHT or
  stream file under manifests/ or results/ is declared in exposure_streams.
- ADM-EXPO judges content, not declarations (spec 20.11; pilot miss INC-HB1-A3): no row of ANY exposure stream,
  feedback_bearing true or false, of the TAUGHT directory or of an undeclared stream-like file may carry a held item's
  answer together with its prompt or orbit, whatever the field names (text, hex, base64 or byte lists; prompt then
  answer in one field counts). A string that is itself JSON is parsed; an answer a few non-digit characters after the
  prompt ('3+7 equals 10') or before it ('10 = 3+7') counts; and records are read in stream order, so a record that
  ends with a held prompt followed by a record whose same field starts with its answer is a leak (corpus S84, S85).
  A listening or presentation stream may show held prompts, never their answers. A held orbit fed with its answer is
  S1 (taught) for the scorer and never counted, and a stream file that is not JSONL, JSON or UTF-8 text cannot be
  checked and fails (corpus S65-S67, S86). A stream-named data file outside manifests/ and results/ (data/warmup.jsonl)
  must be declared too.

## 9. Causal reuse for E3 promotion

Promotion (a SEALED_CONFIRM headline, adoption, a pointer) requires UDI LCB > 0 from leave-units-out variant worlds:
- For each base world, a variant world removes the witness support of a dependent half D of the counted held items and
  keeps the support of the independent items I. For SEALED_CONFIRM, D is drawn from the seal-derived seed (purpose 'udi').
- UDI = mean_w [ (acc_base(D) - acc_var(D)) - (acc_base(I) - acc_var(I)) ], with a world-level LCB.
- UDI-01 checks validity: D items become undetermined, I items stay determined, and TAUGHT(variant) = TAUGHT(base) minus
  the removed set. Checker 1.0.0 registers UDI-01 as planned (not yet enforced; tools/minifly_check/README.md), so no
  gate certifies promotion yet and `claim` writes reuse_specificity not_tested even when SCORES shows a UDI value.
- Shortcuts and teaching-as-trigger learners give UDI of about 0.
- Without it the claim is E3 with reuse_specificity not_tested. Construction and a worked example:
  [references/STRATA_WITNESS_UDI.md](references/STRATA_WITNESS_UDI.md).

## 10. E3 cadence and the component bridge

- The cadence counter is computed (`kb.deliveries_since_last_e3`). After 2 or more deliveries that did not open held
  E3 (CONSTRAINTS.e3_cadence_max), the next lock must open it or carry a UD with scope `e3_waiver:<delivery>` (LOCK-09).
- E3-opening means: primary pool HELD, counted strata within S2..S4, claim_level_target E3, and an official run
  planned with MAIN or SEALED_CONFIRM.
- The counter resets only on an admitted, scored held-E3 primary (S2+ counted) on MAIN or SEALED_CONFIRM worlds. An
  honest exit such as CANNOT_TEST_CLAIM_AS_SPECIFIED does not reset it (corpus fixture C09b).
- Any E0/E1 work states, in its lock predictions, its predicted effect on held E3 (which later endpoint, which
  direction, how large). A component result never stands in for E3.

## 11. Lineage and purity

- Declare lineage.parent (`<dir>@<sha>` or null), changes_declared and priors_added. The checker computes the family;
  an undeclared parent with native-closure Jaccard of at least 0.5 is a LOCK-13 FAIL.
- Native purity scans the closure of files.native and files.runner (LOCK-07): CONSTRAINTS.native_forbidden imports,
  calls or identifiers FAIL; an evaluator module reachable from native or runner FAILs; supplied-arithmetic heuristic
  hits are REVIEW_REQUIRED (WARN at T1; at T2 a REVIEW finding must address them).
- The closure must be complete (LOCK-05). A literal `sys.path.insert` into a sibling delivery is a collision FAIL; V6-R1
  lost a technical cycle when evaluator `import data` resolved to an inherited module after the native bootstrap
  modified sys.path (`minifly_v6_r1_dev_20261023/cycles/01_REVIEW.md:7-9`).
- answer_path must be native for any counted item.
- priors_added appear in the headline. More than one declared factor change labels the claim integrated_intervention.

## 12. OC for confirmation

Before a SEALED_CONFIRM or adoption run: `MFC oc <dir>`. It runs 2,000 Monte Carlo replications (binomial counts per
world at p_comp and p_comp + target_delta, with the between-world SD from the dev d_w or declared) and reports
P(pass | target) and P(pass | null). P(pass | target) must be at least CONSTRAINTS.oc_min_power (0.8), or the run
cannot be SEALED_CONFIRM: worlds_role is capped at MAIN (OC-01). Corpus fixture C21, built from V6's two-unit execution
spec, has OC 0.41 with 4 worlds and must fail OC-01.

## Not required

- No UDI unless E3 promotion is intended or a udi block is declared.
- No OC unless SEALED_CONFIRM or adoption is intended.
- No hand-assigned strata, determinacy flags or shortcut values: the checker computes them.
- No delivery verdict, summary or reducer code.
- No predictions on admission measurements.
- No E3 claim language for E0/E1 work; report the level earned.

## Failure classes covered

FM-02 undelivered manipulation; FM-03 confounded intervention; FM-04 non-discriminating gate; FM-05 untestable testbed;
FM-06 weak comparator or guard; FM-07 proxy metric; FM-11 holdout reuse; FM-12 exposure leakage; FM-16 technical vs
behavioral readiness; FM-17 directional or fixture mismatch; FM-21 statistics; FM-23 lineage and answer-path drift;
FM-30 meta-loop Goodhart (learner level). Definitions: `PROJECT_DOCUMENTS/KB/FAILURE_CLASSES.json`.
