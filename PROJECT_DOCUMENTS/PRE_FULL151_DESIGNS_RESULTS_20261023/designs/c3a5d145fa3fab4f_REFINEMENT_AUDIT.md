# V9.5 — design audit and binding refinements (before implementation)

**Date:** 2026-09-25. **Reviewed:** `DESIGN.md` (Codex, 22:35), `AUDIT.md`, V9.4's `DESIGN.md`, `interference_assay.py` and receipts, V9.3's `joint_assay.py` helpers and V9.2's `brain_v92.py`.

**Status:** these amendments bind the implementation in `run/`. Where they differ from `DESIGN.md`, they take precedence; otherwise `DESIGN.md` applies unchanged. This file and `DESIGN.md` are both hashed into the V9.5 source lock.

## Verdict on the paper design

The causal structure is sound:
- one sealed world bank;
- identical byte, reward and clock streams across arms;
- only the native write flag varies;
- same-prestate shadows for incremental effects;
- a final 2×2 for last-cohort and history effects;
- two terminal readouts under a Bonferroni family;
- learning and floor gates that stop a null result being read as protection.

The arithmetic also checks out: 1,008 arm-records per world, and a 64.5-hour simulated span.

I found one invalid analysis, several ambiguities the runner has to settle, and one requirement I cannot meet as written. None of them changes the question, the sample, the arms, the primary estimand or the decision labels.

## Findings and amendments

| # | Finding | Amendment |
|---|---|---|
| A1 | **Invalid exploratory split.** `DESIGN.md` carries over V9.4's "shared-symbol-containing versus unshared old pair" split. In V9.5, every one of the three old symbols is shared with at least one new cohort, so **every old pair contains a shared symbol** and the split is empty. | Replace it with an **overlap load** per old pair: the number of new cohorts that share either of its two symbols, which is 2 or 3 under the (2,1,1) assignment. Report the W−N effect by load, averaged within each world before intervals are formed. This is exploratory only. |
| A2 | **Probe roster left open.** Which sets are probed in which arm at which time is not specified. | At each E_j: **W** and **N** are probed on old + new cohorts 1…j; **S_j** on old + cohort j; **CLOCK_ONLY** on old. At E4 and D4: all four final arms (WWWW, WWWN, NNNN, NNNW) on old + cohorts 1–4, and CLOCK_ONLY on old. The old arms are probed at the old start, old end and P. The +24 h probes at P use the main-state flush; at D4 they are passive, on copies, as in V9.4. |
| A3 | **"Independent" code review cannot be satisfied.** The implementer (Claude) is the only reviewer available, and I am not spawning a separate reviewer. | This run is labelled **implementer-audited**, not independently audited. The substitute requirements are A4, a negative control for each fatal check (A5), and the written self-audit in `run/CODE_AUDIT.md`. The report states this deviation. |
| A4 | **No check against a known-good result.** Nothing ties the new branch machinery to the already-audited V9.4 runner. | **Regression fixture.** Using V9.4's own `world_for` and `records_for`, run the V9.5 engine with old acquisition plus a single cohort on V9.4 science world **305**, an ordinary world. It must reproduce V9.4's saved receipt bit-for-bit: all old and new pair margins for write and no-write at bout 12 and at +24 h. Any difference is fatal before any V9.5 world runs. |
| A5 | **Fatal checks listed without negative controls.** | Fixtures prove that each check can fail: a swapped receipt, a changed schedule hash, a changed write-flag sequence, a mutated probe, and a clock offset each must be rejected. |
| A6 | **Robustness settings unspecified.** | Bootstrap: 10,000 whole-world resamples with seed **20260926**, percentile intervals at the same 97.5% (primary) or 95% (gates) level. Leave-one-world-out: report the number of worlds whose removal changes a primary label. |
| A7 | **The write-flag ledger is not defined operationally.** | The runner wraps every native `Teach` call and records the arm's write flag. For each arm it stores the ordered flag list's SHA-256, the `write=True` count and the `Teach` count, and checks them against a schedule regenerated independently from the branch graph. This covers discarded shadows too. |
| A8 | **Sentinels run on the science worlds but the technical text check is unspecified.** | Technical world 9901 runs the Doyle/Stevenson 20,000-byte continuations at prechallenge and after cohort 4; 9902–9904 do not. Science sentinels 401–404 run as `DESIGN.md` specifies. |
| A9 | **Concurrency.** `DESIGN.md` allows up to four workers. | Four workers, fixed. Every world is deterministic and single-process, so equality across worker counts holds by construction. The technical gate records wall time and peak RSS. |

## What does not change

The following are all unchanged from `DESIGN.md`:
- the parent checkpoint;
- the namespace `BYTE-CORE-V9-5`;
- science indices 401–656 (n = 256) and technical indices 9901–9904;
- symbol drawing and the (2,1,1) shared-symbol assignment;
- 14-byte records, Δ = 30/14 s and 165 s record spacing;
- 12 bouts per set;
- contiguous cohorts;
- the branch graph;
- the primary estimand (WWWW − NNNN old, at D4);
- the two primary readouts with 97.5% t intervals;
- the labels;
- the three interpretation gates;
- the optional 5-pp choice bound;
- secondary contrasts, noise sensitivities, sentinels, and failure and retry rules.

The teaching dose, topology, readout and byte predictor are not modified.
