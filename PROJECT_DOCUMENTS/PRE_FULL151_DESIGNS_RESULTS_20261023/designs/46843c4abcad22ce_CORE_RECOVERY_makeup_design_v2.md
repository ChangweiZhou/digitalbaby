# CORE-RECOVERY Make-Up Experiment — Refined Design v2

**Date:** 2026-09-17  
**Scope:** repair the historically contaminated late evolutionary search without restarting MiniFly.

## 1. Frozen objective

Intelligence is treated as the efficiency with which experience becomes competence.
The evaluator must reward useful acquisition, useful persistence, revision, anomaly recovery, and recovery when demand returns. It must not reward blanket retention merely because a memory is old.

## 2. What historical reanalysis already resolved

No new simulation is required for V18, V19, V35, V40, V44, V50, V51, V68–V73, or V75–V78 for the endpoint-misalignment issue. V86 remains a bridge/stress assay. V87 CORE-ID remains valid conditional on its historical V84E anchor.

Three historical questions remain scientifically live:

1. OMNIBUS `C_FORGET` — a historical false-negative candidate under blanket retention.
2. V74 `CONTRACT_0.05` — unresolved because competence improved while generic old memories deteriorated.
3. V79E–V85E evolution — must be rerun because old-memory objectives/constraints affected reproduction and search trajectory.

The supplied Python runner implements **(3)**, the expensive contaminated component. The two historical mechanisms should remain separate source-faithful bridge experiments; they are not semantically ported into the late 17-gene genome.

## 3. Why v2 reuses the V88 evaluator

The earlier design proposed a new D1/D2/D3 world generator. That would make the make-up experiment change two things simultaneously: the evolutionary objective and the task generator.

Instead, v2 reuses the already-audited V88 relevance-world machinery, which already measures:

- P1 current usefulness;
- P2 relevant persistence after intervening writes;
- P3 revised relevant persistence;
- P4 revision;
- P5 acquisition;
- P6 anomaly recovery;
- P7 reactivation after four renewed exposures.

The make-up evaluator additionally makes **reactivation exposure 0** a primary competence measurement because it is the learner's first behavior when demand genuinely returns.

Unused collateral retention remains diagnostic only.

## 4. Evolution objectives

To avoid another high-dimensional proxy frontier, the runner uses three direct competence aggregates:

### D1 — learn and maintain useful information

`mean(P1, P2, P5)`

### D2 — adapt correctly

`mean(P3_revised_persistence, revision_curve_mean[1,2,4,8,12], P6_anomaly_recovery)`

The full correction curve replaces a terminal-only revision score so that learning speed matters.

### D3 — demand returns

`mean(reactivation_accuracy[0,1,2,4])`

First-use behavior therefore counts, but it is evaluated together with relearning efficiency rather than becoming a blanket dormant-memory gate.

NSGA-II minimizes `(1-D1, 1-D2, 1-D3)`.

No unconditional old accuracy, old-loss increase, worst-old damage, or collateral survival can enter objective values, constraints, ranking, crowding, or tie-breaks.

## 5. Search counterfactual

The purpose is to change the **evaluator**, not the genome or optimizer.

Therefore the runner preserves the audited V84E defaults:

- same 17 genes;
- same gene boxes;
- population 64;
- 50 generations;
- evolution seed 8410001;
- mutation sigma 0.8;
- mutation probability 0.5;
- random injection 0.1;
- tournament size 3;
- reference genome fixed at zero.

The V81/V84 old-memory performance constraints are removed. Only numerical/source/information-boundary validity remains hard.

## 6. Rotating development panels

A fixed six-world panel across 3,200 candidate evaluations would invite evaluator overfitting.

The 50 generations are divided into 10 five-generation epochs. Each epoch receives six fresh V88 relevance histories. At every epoch boundary the incumbent population is re-evaluated on the new common panel before it produces the next generation.

V88's original seed helper has only historical A/B namespaces, so the make-up runner does **not** rely on arbitrary round-name strings for freshness. Development, validation and confirmation use disjoint campaign-global V88 history indices; this was added after validation caught a seed-namespace collision risk. Raw cue keys are also forbidden from reuse across campaign panels.

This yields approximately:

- 50 evaluated generation populations;
- 9 incumbent re-evaluations at panel changes;
- 59 × 64 × 6 = 22,656 candidate-world lifetimes.

The exact count can differ only through explicit command-line overrides recorded in the frozen manifest.

## 7. Validation and confirmation

After generation 50, a diverse corrected archive is formed from epoch elites and the final Pareto population.

Up to 16 unique candidates are evaluated on 16 fresh validation histories.

The confirmation roster is then frozen as:

- the reference;
- up to three corrected validation Pareto/crowding candidates.

Confirmation uses 96 fresh relevance histories by default. This is a planning value frozen before confirmation; users may override it before the search, but it is recorded in the manifest and cannot be changed in-place afterward.

## 8. Source integrity

Before evolution, the runner requires a native qualification record tied to:

- V88 runner hash;
- V82E source hashes locked by V88;
- iteration23/24 parent source hashes when present;
- V84E config digest;
- Python and NumPy versions.

Qualification checks:

- native reference execution;
- repeated-run determinism;
- spawn/serial equality of the reduced scientific record;
- valid P1–P7 plus full revision-curve and 0/1/2/4 reactivation-curve extraction;
- absence of collateral metrics from selection.

## 9. Resume semantics

Every generation plan is frozen before evaluation. Candidate-world receipts are atomic and reusable. On resume, a previously frozen generation plan must equal the plan reconstructed from the frozen prior scores; otherwise the run stops.

Creating a file named `STOP` in the run directory stops between generations while retaining sealed receipts.

## 10. What the runner intentionally does not do

The runner does **not**:

- rerun every historical version;
- invent a modern `C_FORGET` port;
- invent a modern V74 contraction port;
- call a zeroed gene a pathway deletion;
- perform automatic minimal-core promotion;
- claim that a post-training lesion proves learning-state dispensability.

After confirmation it emits `REDUCTION_HANDOFF.json`. True train-from-zero state/pathway deletion should be a separate V87-style experiment on the corrected finalist.

Resource/state inventories are recorded on every candidate-world receipt and summarized diagnostically. They do not become another evolutionary proxy objective because the 17-gene make-up search keeps the bounded architecture/state-array family fixed; actual state/path removal belongs in the reduction stage.

## 11. Commands

```bash
python3 core_recovery_makeup_runner.py selftest

python3 core_recovery_makeup_runner.py preflight \
  --project-root "$HOME/Downloads/second_try_copy"

python3 core_recovery_makeup_runner.py qualify \
  --project-root "$HOME/Downloads/second_try_copy" \
  --out "$HOME/CORE_RECOVERY_QUALIFICATION"

caffeinate -i python3 core_recovery_makeup_runner.py evolve \
  --project-root "$HOME/Downloads/second_try_copy" \
  --qa "$HOME/CORE_RECOVERY_QUALIFICATION/QUALIFICATION.json" \
  --workers 2 \
  --out "$HOME/CORE_RECOVERY_CORRECTED_EVOLUTION"

caffeinate -i python3 core_recovery_makeup_runner.py confirm \
  --project-root "$HOME/Downloads/second_try_copy" \
  --workers 2 \
  --out "$HOME/CORE_RECOVERY_CORRECTED_EVOLUTION"
```

Run `qualify` successfully before launching `evolve`.
