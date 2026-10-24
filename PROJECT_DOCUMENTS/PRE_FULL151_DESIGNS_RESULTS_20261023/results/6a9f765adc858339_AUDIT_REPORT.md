# V9.5 independent receipt re-audit and compact evidence export

Date: 2026-09-26. This is a read-only analysis of the frozen V9.5 run. No learner was rerun, no threshold was changed, and no original source, receipt, or report was edited.

## Verdict

The V9.5 result survives the receipt audit. Across 256 science worlds, four successfully learned new byte cohorts cause a persistent cost to old associations relative to an identical no-new-write history:

| Old-set endpoint at D4, WWWW − NNNN | Recomputed mean | 97.5% paired t interval |
|---|---:|---:|
| Deterministic choice | −11.328 percentage points | [−14.489, −8.168] pp |
| Analog margin | −0.022711 Hz | [−0.026371, −0.019051] Hz |

The old-learning, four new-cohort-learning, and no-write-baseline gates also reproduce. The five-set D4 competence label survives. These claims apply to one model seed, one fixed teaching dose, and the specified overlapping byte workload; they do not identify a synaptic, compartmental, or capacity mechanism.

## What was independently verified

- All **256 science and 4 technical receipts** were read from the existing run. The archived source-lock digest reconstructs, and all **16 locally available named source files** match their frozen SHA-256 hashes.
- Each receipt's world symbols, shared-symbol assignment, old-pair overlap labels, ordered-byte training schedule hashes, timing, branch accounting, probe roster, and recorded old-pair loss counts were checked. The scalar choice, mean margin, pair choices, minimum margin, and both noisy-choice scores were reconstructed from every saved three-pair margin vector.
- The stand-alone reducer recomputed the primary endpoints, bootstrap intervals, gates, final arm readouts, secondary contrasts, pair losses, worst-pair effect, and other saved summary tables. **1,167 comparable numeric fields match the frozen summary exactly**; there are no missing comparable fields or mismatches. For KC overlap, the reducer checks aggregation of the saved overlap metadata; it cannot regenerate KC codes from the compact evidence alone.
- Eight deliberate synthetic corruptions are rejected by the new reducer, including an impossible choice, NaN margin, wrong overlap label, wrong pair-loss count, and missing S2 shadow. A direct check of the frozen validator confirmed that it accepts a choice of 2.0, a NaN scalar margin, and an impossible `[99,99,99]` overlap label. These are validation defects, **not evidence that any actual V9.5 receipt is corrupt**.

The original validator and report remain frozen because editing them would invalidate the run's source lock. This separate audit supplies the missing assurance for the existing receipts; future runners should validate all derived score fields before aggregation.

## Correction to the rest interpretation

The WWWW-minus-NNNN *choice contrast* narrows between immediately after cohort 4 (E4) and 24 hours later (D4). The WWWW old-choice mean does not improve:

| Arm | E4 old choice | D4 old choice | Paired D4 − E4, 95% t interval |
|---|---:|---:|---:|
| WWWW | 61.719% | 61.589% | −0.130 pp [−3.606, +3.345] |
| NNNN | 77.604% | 72.917% | −4.688 pp [−6.693, −2.682] |
| WWWW − NNNN | −15.885 pp | −11.328 pp | +4.557 pp [+0.850, +8.265] |

Thus the relative choice deficit narrows mostly because the no-new-write arm loses its temporary advantage. The analog write deficit moves in the opposite direction: its paired D4-minus-E4 change is **−0.004555 Hz [−0.007053, −0.002056]**. These post-rest intervals are additional descriptive analyses from the existing worlds, not prespecified new primary endpoints. Individual old pairs can improve even though the all-write mean is flat.

The pair-loss counts also reproduce: 563 of 768 old pairs were correct just before the challenge; by D4, 123 had been lost under WWWW versus 3 under NNNN.

## Claims that remain unresolved

- Old-pair overlap load 3 minus load 2 has a D4 paired margin-effect difference of **+0.000624 Hz [−0.007773, +0.009021]**. This does not establish absence of an overlap gradient or equivalence.
- The D4 choice interaction is **+0.130 pp [−2.577, +2.837]**. Its direction is unresolved; the data do not show that sub-additivity is definitively absent.
- The policy contrast measures the total effect of new writes along the subsequent learner trajectory. It does not directly measure erased information or isolate the anatomical site of interference.

## Audit artifacts and reproduction

- `independent_reducer.py`: stand-alone read-only validator and reducer. It imports NumPy and SciPy but does not import or run the V9.5 learner.
- `verify_compact_export.py`: rechecks primary intervals, bootstrap, acquisition gates, rest changes, and pair losses from the portable evidence alone; it needs no full receipts or learner code.
- `results/PER_WORLD_EVIDENCE.jsonl`: compact evidence for 256 worlds: pair identities, old/new pair-margin vectors at prechallenge and E1–E4/D, same-prestate shadow margins, source/schedule hashes, and timing/write ledgers. It excludes event streams, model arrays, and full receipts.
- `results/RECOMPUTED.json`: all recomputed statistics, source and receipt hashes, synthetic corruption checks, and frozen-summary comparison.
- `results/TECHNICAL_EVIDENCE.json`: technical-gate summary and the four technical receipts' checks and ledgers.
- `frozen_validator_probe.py` and `results/FROZEN_VALIDATOR_PROBE.json`: reproducible synthetic check of the original validator's omissions.

From the project root, run `python BYTE_CORE_V9_5/audit_20260926/independent_reducer.py`. It reads the frozen `BYTE_CORE_V9_5/run/results` directory and overwrites only this audit folder's generated `results` files. The compact package's manifest points back to the unchanged full receipts and parent assets for deeper review.
