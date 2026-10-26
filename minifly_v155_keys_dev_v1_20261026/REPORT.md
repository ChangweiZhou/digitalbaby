V155 KEYS V1: DEV_NOT_EVIDENCE. Stopped at the C1 decision point: L-PRED CANNOT_TEST_CLAIM_AS_SPECIFIED; the owner chose option A, carried out as V158.

# V155 report: local learning of MiniFly's keys (26 October 2026)

**Status:** C0 testbed CONTINUE. C1 technical closed as *escalate*: every formula and integrity check passed, but the pre-stated fixture failed. No dev, rehearsal or official run took place.
* **Design:** `minifly_v155_keys_design_20261026/DESIGN.md`. §12 holds the pre-lock refinements, §12.9 the reference revision after synthetic qualification, and §12.10 the L-PRED correction made before any learner code.
* **Grant:** `AUTHORIZATION.json` (default caps).
* **Ledgers:** `runlog/ATTEMPTS.jsonl` (5 attempts), `runlog/CYCLES.jsonl` (testbed, technical), `runlog/AMENDMENTS.jsonl` (1, operational).

## C0 testbed (*Thursday*, 40 KB taught; held-out bits/symbol)
| Arm | Held-out | Note |
|---|---:|---|
| FROZEN: birth keys, R1 on the binary code | 2.963 | equals V153 and the headroom check |
| PERM: birth weights permuted over the same synapses | 3.030 | 0.067 worse (SD 0.011): the birth arrangement carries information |
| REF-N: free graded layer on the native synapses (back-propagation, evaluator only) | 2.931 | 0.032 better (SD 0.014); best epoch 59 of 60, so a lower bound |
| REF-D: the same with dense connections | 2.740 | 0.223 better (SD 0.015) |

* The verdict was CONTINUE: parity, ceiling and floor were met, and neither reference was stopped by the budget.
* The cached code path reproduced MiniFly's own encode at all 56,426 positions.
* **One operational failure:** attempt testbed-002 crashed in the runtime-closure helper after writing its results. torch reports a relative `__file__` for `torch._classes`. The helper was fixed and the lock rebuilt (amendment 1). The step was re-run and reproduced every number exactly. The failed output is kept and was not used.
* **Before the lock, synthetic checks showed** that back-propagation through the binary top-k layer cannot learn keys reliably: it recovered about 0.1 of a known 0.5-bit headroom (`prelock/OUTPUTS.md`). The references were therefore made descriptive (design §12.9).

## C1 technical
* **9 of 9 formula, integrity and determinism checks passed.**
  * Both rules' single updates equal independent dense computations (maximum difference 2.2e-16).
  * Column sums, signs and wiring are kept.
  * The shared key data stays consistent.
  * Runs are deterministic.
* **The fixture of design §12.10 failed.**
  * It uses synthetic streams in which bytes of a set share their futures and have identical pasts.
  * The index is the mean code overlap for different bytes of the same set, minus that for bytes of different sets.

  | Stream | FROZEN | L-COMP | L-PRED |
  |---|---:|---:|---:|
  | seed 0 | 0.0042 | 0.0029 | 0.0019 |
  | seed 1 | 0.0050 | 0.0038 | 0.0033 |

  The rule required L-PRED ≥ FROZEN + 0.01 on both streams. L-PRED is therefore CANNOT_TEST_CLAIM_AS_SPECIFIED.
* **Cost:** L-COMP 0.53 ms per position, L-PRED 5.6 ms per position (about 10× the design's estimate).

## Diagnosis (analytic; tested in V158)
* Both rules update only units that are already active, moving their weights towards the current input. That raises each unit's selectivity for contexts it already responds to, so overlap between different contexts can only fall. Both rules lowered the index.
* For contexts with a common future to come to share KCs, a unit that is predicted (here, backwards from that future) but currently inactive must be able to strengthen its synapses from the current input. Only then can it be recruited. The derivative surrogate chosen in §12.5 (only active units update) excludes that.
* This is the same dead-unit limit that defeated back-propagation through the top-k layer before the lock.

## Owner decision (26 October 2026)
* Options shown:
  * A: revise L-PRED so that predicted near-threshold units can be recruited, and repeat the technical cycle;
  * B: continue with L-COMP only;
  * C: stop and write up.
* Recommendation shown separately: A.
* Owner's reply, verbatim: "Okay, we do option A. Call this V158."

## Accounting
* 1.26 CPU-h of child CPU, 2,534 learner records (all C1 fixture and technical runs), 1 of 3 test cycles, designs tried k = 2.
* Pre-lock synthetic checks: about 0.25 CPU-h, unlaunched, no text.
