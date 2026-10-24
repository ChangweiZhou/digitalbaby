# MiniFly: an action-first route to `1+2=3`

**Status: alternative design for hostile review, 26 September 2026. No experiment is authorized by this document.** It does not amend the frozen V2 result, W3, W4, Full151, or any earlier report. The first implementation would be an interface and development pilot only. A science run needs a separately frozen candidate, fixture, runtime receipt, acceptance script, and independent audit.

## Why change course

The goal is a learner that receives bytes, uses earlier experience, and produces a correct answer itself. The [project objective contract](MINIFLY_OBJECTIVE_CONTRACT.md) distinguishes directly taught association (E1) from reuse on never-reinforced inputs (E3). V9.9A/B found modest native E3 effects on synthetic pair choices, but the evaluator supplied both alternatives. V2's arithmetic candidate chose an integer that a wrapper printed; its sealed one-operator exact score after 512 graded questions was 1.3%, versus 7.7% no-write. W4 proposes four operations, several new write/read mechanisms, evolution, and persistence in one program. This document instead asks which link in one short addition-and-output chain fails. It is a scope reduction, **not** a shorter execution of W4 or evidence that the existing native circuit already performs arithmetic.

## Two claims that must not be merged

1. **First autonomous output (E1):** after direct teaching of `1+2=3`, can one continuing learner read `1+2=` and emit the bytes `3;` before feedback? Passing shows learned recitation plus an output path. A lookup table could pass, so this is not arithmetic competence.
2. **First unseen addition (bounded E3):** after training on other sums but never teaching either orientation of the pair `{1,2}` *within that life*, can the learner emit `3;` for both `1+2=` and `2+1=`, and answer other never-taught operand pairs? The `1+2` example is a **public demonstration item**, not a blind statistical confirmation: Gate 1 and this document already expose it. The scientific claim must depend on separate, program-level untouched pairs, on prior teaching state, and on gains beyond lookup, output-frequency, and permuted-teacher controls. Delay and intervening learning are a later gate; until they pass, do not call the result a persistent core.

Neither claim means spontaneous conversation. The newline and `=` are explicit task cues. Removing those cues is a later interface question, not a hidden promise of this study.

## Fixed first task and byte contract

Use only **single-digit nonnegative addition** in the first program: operands `0…9`, answer `0…18`. No subtraction, multiplication, division, precedence, spaces, or multi-digit input operands. This makes the question at most four meaningful bytes, but a previous two-byte LAT4 result does **not** certify this whole expression. A question is delivered incrementally as `\n` then ASCII `a+b=`. After `=`, the learner must choose each output byte through its own output state: one or two ASCII digits followed by `;`. The environment neither enumerates candidate answers for a value comparison nor converts an integer choice into text. At most three output bytes are allowed; missing terminator, malformed output, or a wrong number is incorrect. A scripted oracle may exercise the harness but must never be part of a scored learner.

In answer-revealing training only, the environment records the complete **first** response and subsequently sends `#<correct decimal answer>;` as ordinary timed input bytes. Correct-answer construction is evaluator-side and must occur after first-response recording. No correct answer, grade, or target token is supplied at a scored probe. The candidate's teacher update must name its actual data dependency: if raw answer token IDs are used directly as supervised targets, call the result **supervised engineering**, even if the same bytes were also presented to the sensor. A claimed sensory-teaching route needs a lesion that holds target IDs and timing fixed while disconnecting the teacher's sensory activations. A claimed native Full151 learning contribution needs native write **and** native read lesions. Keep the byte-only engineered comparator, a MiniFly-input-plus-new-head candidate, and any native-writable candidate separate in every report.

The visible newline may clear expression-local working state. It may not clear learned cross-question state. The output cannot inspect hidden operands, parsed integers, the evaluator's answer, or future teacher bytes. Probes are from full-state clones at one common time; verify the continuing training state hash is unchanged. Log model state, optimizer state if any, clock, and RNG in each clone and restart receipt.

## Gate 0: prove access and output before asking for learning

Run no science worlds. Unit and fixture checks must demonstrate:

- A scripted oracle, using the **same incremental byte and output API**, can receive every allowed expression and emit its exact answer bytes. This checks the harness, not MiniFly.
- The candidate's state after `=` retains information about **both** operand values and the presence of `+`, sufficient to distinguish questions with different answers. A representation may treat `1+2` and `2+1` identically: addition is commutative, so recovering their presentation order is not an entrance requirement. Inspect a fixed, prospectively chosen decoder and an appropriate random-label control; a post-hoc decoder cannot qualify the interface.
- Its output circuit can emit every digit `0…9`, a second digit when needed, and `;` under controlled non-arithmetic states. No wrapper may repair its output.
- Substituting any future teacher bytes cannot alter the already recorded first response; scored probes cause no cross-question write; save/load and uninterrupted/resumed transcripts match.
- The actual selected byte timing and full 5,177-KC versus pooled/full-route choice are recorded. FE0-at-1-second with 128-bin pooling cannot silently stand in for an untested full route.

Gate 0 failure stops arithmetic work and identifies an interface defect. It does not justify a population search.

## Gate 1: the smallest learned-action test

Teach `1+2=3` repeatedly with answer bytes after each first response. On a subsequent read-only `1+2=` probe, compare the same initial candidate and input schedule under true teacher-dependent writes, identical teacher bytes with all relevant cross-question writes disabled, and wrong teacher bytes of the same length. Record **exact emitted bytes**, not a value margin. Include other taught expressions so emitting `3;` unconditionally cannot pass the whole E1 fixture. An independent key→answer table may verify that the harness can convey arbitrary answer bytes, but is not a mandatory entrance exam for every arithmetic candidate.

If true teaching cannot improve first-response output over its controls on directly taught expressions, stop. Do not widen to held-out arithmetic, change topology, or evolve hyperparameters on the basis of this failure alone. An E1 pass permits Gate 2 development; it is not a science claim about addition.

## Gate 2: unseen sums on a small, inspectable fixture

For a first **development** fixture, unordered operand pairs are the exclusion unit. There are 55 pairs `{a,b}` with `0≤a≤b≤9`. Reserve the following 11 pairs, **both orientations**, for no-feedback development probes:

```text
{1,2}, {2,3}, {3,4}, {4,5}, {5,6}, {6,7},
{0,4}, {1,7}, {2,8}, {3,9}, {5,9}
```

**Before any development learner is trained**, an independent fixture-lock step must additionally reserve 11 *different* unordered pairs from the remaining 44 as a sealed confirmation roster. It must withhold their identities and answers from code and people selecting a model. The 33 pairs left after both reserves supply the development training expressions; a feasible partition with 56 ordered training expressions exists, with every digit and every reserved answer value represented in training. The exact final roster and its hash are produced only by that fixture-lock step, not retrofitted after development. Both orientations of each unequal reserved pair are excluded from **all** teaching and grading in **all** development and confirmation lives. If the lock cannot satisfy these support and disjointness checks, stop before training. Repetitions of allowed training expressions are counted; no reserved pair is repeated with feedback. The public 11-pair development fixture is consumed for model selection and cannot later be described as sealed confirmation.

From independently initialized lives, score every development-held-out item on a separate read-only clone **before any feedback on that item**. Descriptive quantities are exact emitted-byte count out of 22, the two public `1+2` orientations, per-pair results, invalid-output rate, and paired true-teaching minus no-prior-write and permuted-teacher differences. Compare a constant-output policy, answer-frequency mode, raw/canonical/reversed-pair lookup, and a simple symbol-feature shortcut trained on precisely the same examples. A symbolic adder is only a harness ceiling. Permuted teacher answers must preserve answer length and the observed answer multiset while assigning every training question an incorrect answer; validate that matching before a run. Keep the same questions, clocks, and output policy across paired arms. If a learner changes its own actions and hence later feedback, label that comparison on-policy; use a separately named identical-history replay for a write-mechanism contrast.

Before **any** held-out science claim, a small positive-control learner must show a reproducible causal improvement on development-only addition fixtures under the same example budget. It may be engineered, but its result cannot be attributed to Full151. If this control fails, revise the curriculum, representation, or example budget on development data. If it passes while the fly candidate fails, lesion input, teacher update, and output one at a time. Do not search thousands of genomes to fill an unresolved access or teaching gap.

## Later gates, only if Gate 2 works

Use only the **already reserved, untouched** confirmation roster, after an independent code check of pair exclusion, answer support, event timing, shortcut performance, and same-life causal comparisons. A single `1+2` success is never a statistical conclusion. Before opening the final roster, freeze a joint per-life rule requiring useful accuracy across its entire blinded set and a positive true-teaching effect versus both no-write and wrong-teacher arms. The public `1+2` and `2+1` demonstrations are reported separately, including failures; they cannot rescue a failed blinded result. Report uncertainty by learner life and by scored pair; repeated lives on one roster are not independent mathematical facts. A failed final roster is reported as failure, not repaired or reused for another “final.”

Only after confirmed immediate E3 behavior should one clone wait, learn a disjoint new cohort, wait again, and probe the old held-out sums before feedback, paired with a no-prior-teaching state. Grade-only learning, multi-digit operands, other operators, irregular byte timing, removal of `=`, and spontaneous output each need separate contracts. Do not bundle them into the first addition gate.

## Attribution and stopping rules

Passing through an engineered byte predictor or new output head shows a working **MiniFly-adjacent behavioral path**, not that the inherited Full151 writer learned arithmetic. Full151 attribution requires a positive same-history effect of native prior writing and a separate native-read lesion, while the answer still comes from the learner's emitted bytes. If the engineered control passes and the native route does not, preserve both findings. If only direct `1+2` teaching works, report E1 recitation. If held-out sums work immediately but fail after later learning, report bounded E3 without persistence. If the arithmetic rule emerges only through an explicitly parsed or hard-coded adder, report an engineered calculator, not learned arithmetic.

No population search or large confirmation follows automatically from a gate failure. Each diagnostic job contains **whole lives**, because no qualified mid-life cursor is assumed. Time one complete life including controls, cloned probes, hashing, and audit before choosing packet count; target approximately one hour per job, but do not assert a hard 60-minute bound without checkpoint support. Keep only source, design, small receipts, aggregate results, audit, and asset hashes; do not duplicate old trajectories or large event ledgers.

**Next authorized work if this design is accepted:** independent hostile review of the byte/teacher attribution, feasibility of the 11-pair fixture and its wrong-teacher matching, and the proposed control and confirmation separation. No arithmetic experiment, native adapter, or evolution run is authorized merely by shipping this document.
