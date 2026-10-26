V158 RECRUIT V1: CANNOT_TEST_CLAIM_AS_SPECIFIED at the C0 testbed (synthetic streams; no learner code was written)

# V158 report (26 October 2026)

**Question:** can a predictive local rule, revised so that near-threshold KCs can be recruited, make contexts with a shared future share KCs?

**Answer:** in this system, not by learning keys alone. The testbed showed that the competition between KCs rejects shared units even when they are built by hand. By the rule fixed in the design (§2, §4, §7), V158 stopped before any learner code. The revised rule L-PRED-R was not implemented or run.

* **Design:** `minifly_v158_recruit_design_20261026/DESIGN.md`.
* **Grant:** `AUTHORIZATION.json` (option A, default caps).
* **Ledgers:** `runlog/ATTEMPTS.jsonl` (2 attempts), `runlog/CYCLES.jsonl` (testbed, decision *stop*), `runlog/AMENDMENTS.jsonl` (1, operational: our `fixture.py` shadowed a MiniFly module of the same name and was renamed `samefuture.py`; the first attempt failed at import before computing anything).

## Result
The streams are V155's fixture: two seeds, 12,300 positions each. Bytes of a set share their futures, and their pasts are identical in distribution. The index is the mean code overlap for different bytes of the same set, minus that for bytes of different sets.

| Stream | FROZEN (birth keys) | ORACLE keys | ORACLE-CODE |
|---|---:|---:|---:|
| seed 0 | 0.0042 | 0.0039 | 0.482 |
| seed 1 | 0.0050 | 0.0049 | 0.483 |

* **FROZEN** reproduces V155 exactly. The floor (≤ 0.01) is met.
* **ORACLE-CODE** inserts a fixed shared set of 46 KCs per side into FROZEN's codes in each set's contexts. The index rises to 0.48, so the measurement detects sharing when it is present.
* **ORACLE keys** give 92 KCs per set (46 per side) incoming weights equal to that set's mean PN input, scaled to the 95th percentile of birth column sums (1.114). Those KCs were active in **none** of the set contexts (mean 0.0 per position) on both streams, so the index did not move. The ceiling (≥ +0.05) is not met.

## Why (analytic; consistent with V155 and with this result)
* **The weight budget.** Every KC has a fixed total input weight: its column sum, which V155's rules also preserve by rescaling. A KC that responds to *any* of m interchangeable inputs must spread that budget over all of them. It therefore receives roughly 1/m of the drive that a KC specialised on one input receives. Here m = 8.
* **The competition.** The code keeps the 92 most strongly driven KCs on each side, so specialists always win.
* **The consequence.** Under a per-KC weight budget and top-k selection by drive, units shared across interchangeable inputs cannot enter the code, whatever rule shapes the keys. This explains V155's result, where both local rules lowered the index. It also implies that V158's revision (recruiting near-threshold units) could not succeed, because a recruited unit that spreads its weights loses the competition.
* **What this does not cover.** Inputs whose alternatives share receptor channels could still share KCs. Mechanisms that change the competition were not tested: per-KC adaptive thresholds (homeostasis), or a divisive normalisation of each unit's drive.

## What this means for the program
1. **The same-future sharing that NLP generalisation needs cannot come from learning PN-to-KC keys alone** while KCs compete by raw drive under a fixed weight budget. This holds for native and for dense wiring, since the oracle was dense.
2. **Owner decision D3 (new synapses) does not help this particular problem**, because a dense oracle failed too.
3. **The candidates that remain change the competition or the place where sharing is stored:**
   * **Homeostatic thresholds.** FastMemory has a native threshold homeostasis: a KC that rarely wins lowers its threshold. That could let consistently driven but weaker shared units enter the code. It was switched off (thresholds zero) in V149, V153, V155 and V158.
   * **Sharing at another level.** For example, learned associations between codes (V155's prediction matrix W), or the output side.

   None of these is chosen here.

## Accounting
* 0.027 CPU-h (two attempts), 0 learner records, 0 test cycles, designs tried k = 1.
