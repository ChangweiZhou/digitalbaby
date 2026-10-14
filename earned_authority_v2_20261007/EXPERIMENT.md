# MiniFly experience-conditioned read authority — revision V2

7 October 2026. **Reviewed prototype; formal launch is not justified.**

This revises the earned-authority V1 proposal without editing its protocol, code or results. The current deliverable is a precisely specified mechanism, executable prototype, hostile audit and exploratory replay. It is not a released native worker or a completed confirmation experiment.

## The stage goal stays fixed

Improve the bounded byte-memory core's useful reuse per matched knowledge and compute budget while preserving taught acquisition, retention and revision. Keep the existing lifetime, binary-relation and Latin-completion fixtures. The objective remains modest reuse of earlier experience; no NLP, open generation, expanded alphabet, new encoder, bigger brain or intelligence requirement is introduced.

The native learning system remains ERROR: four shared FE0 stores and four private CONTENT stores. Its private teaching uses its own pre-answer probabilities, not the emitted readout action. Consequently an authority mechanism can be replayed on archived native predictions without repeating native trajectories. This equivalence would fail in an environment whose next experience or update depended on the emitted action.

The authority observer is **additional engineered learning state**, not part of the previously measured native Full151 state. Its cost and causal contribution must be counted. A new observer learning from old records is new learning; it must not be described as a purely stateless, zero-learning analysis.

## Why V1 was unsatisfactory

V1's fixed margin window did not earn authority from experience. For ordinary binary conflicts its decisions reduce to Q_HALF, apart from exact ties/boundaries. Its prospective large run would therefore largely repeat the known half-weight question in binary choice.

Replacing that with a global teacher-accuracy gate is also inadequate. Ordinary teaching strongly favors private exact-key recall. Across the archived W histories, half-weighting helped/hurt on 15/1,193 binary teacher predictions and 96/4,897 lifetime teacher predictions. A global gate would learn to retain ERROR even where half-weighting helped later reuse. Margin size is not demonstrated reliability, and ordinary training accuracy is not demonstrated query-conditional utility.

This revision asks one bounded question: **can the learner's own past reinforcement support distinguish experiences on which less private authority is useful?** It does not assume the answer is yes.

## One frozen mechanism

The exact pre-replay rule is `RULE_LOCK.json`; the executable definition is `experience_policy.py`.

1. Keep a 64-entry FIFO of CONTENT sensory addresses: the last four non-space bytes of the current line. It contains presence only, not answer bytes, values, margins, semantic IDs or task labels. Repeated writes do not refresh FIFO order. A 65th distinct address evicts the oldest.
2. Before an ordinary teacher event, capture the current key and classify it from **past permitted writes** as supported or unsupported. Short contexts are invalid for this design, rather than silently treating FE0 fallback as CONTENT.
3. Compute the unchanged ERROR and Q_HALF digit predictions before the answer arrives.
4. After the actual arriving digit, and only if the real teaching flag permits writing, update that stratum's three counters: observations, Q_HALF-only correct (`help`), ERROR-only correct (`hurt`). Insert the address after scoring. A disabled write changes neither counters nor support.
5. At query, use private weight 0.5 iff past `help > hurt` in the current stratum; otherwise use 1. Empty and equal counters retain ERROR. No posterior probability or confidence guarantee is claimed.
6. A two-option query is supported only if both sensory addresses are in the cache. One common weight applies to both options. Combine each option's four normalized channels, then compute channel 1 minus 0, preserving frozen arithmetic and tie behavior.

Normalization remains shared=1.4911274663291492, private=1.3452365735750882. No thresholds, priors, weights or bins were changed after replay. The sign rule is an engineering hypothesis, not an optimized estimator.

The paired query accepts keys and bank observations only. It cannot receive a target, evaluator-taught flag, world, stage or task. The environment's existing sensory protocol still distinguishes one-cue byte prediction from two-option choice. Capturing a supported address from the model's own write history is legal internal evidence; it is not free or native biological familiarity.

## Controls and attribution

All readouts see the same native trajectory and teacher events:

| Readout | Definition | Role |
|---|---|---|
| ERROR | Private weight 1 | Native reference |
| Q_HALF | Private weight 0.5 | Existing fixed-weight reference |
| EXPERIENCE | The two-stratum past-feedback rule above | Sole candidate |
| GLOBAL_CALIBRATION | Pool the two strata's help/hurt | Does locality matter? |
| SUPPORT_ONLY | Half on unsupported addresses, full on supported | Does support alone explain the gain? |

The controls use the same bounded observer allocation; they are attribution readouts, not extra candidates eligible for promotion. Choosing SUPPORT_ONLY after seeing its replay results would be a new design, not confirmation of EXPERIENCE.

The explicit branch map remains W (all writes), N_old (no old writes), N_new (no new writes) and N_revision (no revision writes), in the assays where each applies. Every applicable native store and the authority observer obeys the same real flag. Primary E3 compares W with joint N_old. It measures the **whole integrated learner**, not native Full151 alone.

An explicitly named OBSERVER_ONLY_OLD diagnostic teaches the observer on the native N_old history. It is not the no-old-learning primary control. Its purpose is to expose a benefit caused by authority learning alone. Stored probe labels are used solely to score already emitted answers; they never update authority. Probe queries leave continuing observer state unchanged.

## The important unresolved assumption

The observer learns correctness of a **single-cue four-way digit prediction**. Binary probes ask for a **two-option L/R comparison**. Those are different losses. A digit prediction can improve without the relative comparison improving, and vice versa.

Transferring an authority estimate between these protocols is the hypothesis being tested, not a calibration theorem. We do not manufacture paired teaching targets, update on held-out answers, introduce task-specific bins or give a novelty prior favoring the shared route. None of those would be the same mechanism.

Latin also remains in the assessment. Read arbitration can expose existing useful values; it cannot establish that the required completion information is present. The accessible Latin data cannot justify a favorable novel-query default.

## Evidence, exposure and countermodels

Lifetime measures E1: exact taught acquisition, intact-old retention and revision. The 24 intact old items exclude eight revised answers. Binary and Latin held-out cases can support only their bounded E3 claims if prior learning improves pre-feedback output. No E4 claim is possible.

Native teaching exposure and chronology are unchanged. Authority sees actual past teacher bytes on permitted records; scored held-out cues have no teacher. The 64-key observer adds reinforcement-support information, not correct unseen answers.

An exact or canonicalized answer table can explain taught recall, but has no answer for the withheld relation instances. A support table can detect unfamiliarity but cannot by itself provide the withheld answer. Binary frequency shortcuts and balanced Latin outputs give chance baselines. Generator class tables and Latin tables remain evaluator-only solvability witnesses. An authority benefit that only changes a biased N_old baseline is reported separately from raw W accuracy.

## Test and replay completed in this revision

No new native life was run. Archived inputs were regenerated by the frozen fixture functions, and archived predictions/teacher bytes were replayed chronologically to train new authority observers.

The replay covered 48 previous binary jobs, 48 previous lifetime jobs and the three already completed V1 technical jobs. The Latin contribution is **one development world**, not a 64-world Latin confirmation. These previously inspected worlds are exploratory development evidence and cannot be reused as fresh confirmation seeds.

Exact ERROR/Q_HALF parity held on **191,664 teacher records and 77,464 probe rows**. Authority updates were enabled on 141,888 records and suppressed on 49,776. These counts span multiple causal histories; they are not independent samples or new acquired facts.

Nineteen prototype test methods passed, including pre-feedback sequencing, causal suppression, no query updates, common paired authority, key capture before outcome, aliases, 65-key eviction, snapshot/resume equality, independent cloning, output-permutation behavior away from inherited ties, counter exhaustion and nonfinite arithmetic rejection. Twelve malformed snapshots and eight tampered actual-write receipt cases were rejected. The latter checks use archived instrumented evidence; they are not new live native write-path qualification.

An independent implementation, without importing the live policy, checked 5,040 teacher records, 1,624 query rows and 41 saved observer states across three selected archived jobs. It matched FIFO, counters, emissions and actual native write flags. This is a spot audit of the new replay, not another full validation run of the old experiments.

## What the replay found

| Final binary, 48 archived worlds | W accuracy | N_old accuracy | W−N_old |
|---|---:|---:|---:|
| ERROR | 57.64% | 53.82% | +3.82 pp |
| Q_HALF | 61.11% | 52.43% | +8.68 pp |
| EXPERIENCE | 57.99% | 53.82% | +4.17 pp |
| GLOBAL_CALIBRATION | 57.64% | 53.82% | +3.82 pp |
| SUPPORT_ONLY | 61.11% | 52.43% | +8.68 pp |

EXPERIENCE changed only **one of 288 W held-out responses**: a descriptive +0.35 pp gain over ERROR. At old end it exactly retained ERROR's held-out accuracy. Both the source worlds and policy family have informed development, so these are not fresh inferential claims.

All historical W lifetime final taught endpoints stayed identical to ERROR: intact-old 97.74%, new 99.74%, revised 96.35%. The support cache protects the known-address readout in this fixture; this does not demonstrate an improved native storage mechanism.

At final W binary, unsupported teacher events produced only 15 helps and 13 hurts across 864 first-address presentations; supported events produced 0 helps and 1,180 hurts across 9,504 records. Only nine of 48 worlds selected half weight for unsupported final queries. That decision changed one W answer. The observed effect is weaker than a fixed novelty rule, not evidence of useful calibrated authority.

In Latin DEV, EXPERIENCE preserves taught output at ERROR's 75% old/75% new, and moves W held-out from 0/4 to 1/4. N_old is also 1/4: causal E3 remains zero. This single world neither proves Latin impossible nor supports adoption.

## Resource and interface boundary

The hypothetical packed observer layout is 313 bytes: 256 address bytes, cache length, six uint64 counters and one pending forecast. The actual Python observer measured approximately 3,572 bytes with a full cache, excluding module/runtime allocations. Counter exhaustion is a terminal error, not silent saturation. Query work scans at most 64 fixed entries and four-channel evidence; it does not scale with lifetime history.

Synthetic timing of all five diagnostic readouts was approximately 27 μs per paired query and 20 μs per predict/observe event on this host. This is an observer microbenchmark, not a measured native-system compute ratio or a formal run forecast. Replay used approximately 26 CPU seconds, in addition to testing/auditing.

The CONTENT four-visible-byte window, key aliases, four output channels and supplied two-option choice remain explicit limitations. Existing fixture rosters fit the 64-entry cache; synthetic eviction was tested, scientific performance beyond that support capacity was not.

## Decision and stop

**Technical prototype: pass. Scientific adoption: not demonstrated. Formal run: not recommended for this rule.**

This is a scientific judgment from the exploratory evidence, not a newly invented significant/non-significant futility test. The observer is active in some contexts but does not recover most of the known binary reuse gain; it supplies no credible Latin evidence. Therefore this revision does not release the previous 192-job/576-life plan. It does not tune the rule or silently promote its SUPPORT_ONLY ablation.

Negative evidence applies to this frozen help/hurt-and-support rule, not to all read authority, all core mechanisms or all Latin learners. Before a future integrated-core candidate is allocated fresh science, its mechanism must explain how ordinary experience can improve the relevant reuse behavior while preserving storage. Calibration vocabulary alone is insufficient.

The native-plus-authority worker, joint checkpoint and operations adapter are intentionally not built or qualified after this negative replay. If a genuinely different candidate is later selected, its complete state and disabled-write paths must be qualified before a separately authorized launch. The existing 10-hour limit, 8-worker maximum, attached caffeinate, record-boundary pause and no duplicate native execution remain operations requirements; they do not authorize a run here.

No core version is promoted by this revision. The previous validated core/reference and all frozen results remain intact.
