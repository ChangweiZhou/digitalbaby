# E16-F7 predictions: six more drift seeds and a pre-registered early signature (10 November 2026, written before the births)

Brief: `OPUS_TASK_E16F7_E16M3_CLOUD_JOBS_20261109.md`, Job 1. This session runs the whole job, both halves and all six seeds, so the review paragraph is `E16F7_REVIEW_SECTION_9_62.md`. Harness at commit e78aa83e, the commit the brief ships with; no harness file is changed by this job.

**Before launch** (checked 03:06 UTC, 10 November): no `exp45/e11_dev_M_<seed>.*` existed for 1100009, 1200007, 1300021, 1400017, 1500007 or 1600033, and no `E16F7*` file existed.

## Runs

1. **Births:** the six births, launched together, detached: `python3 exp11_cell.py dev M <seed>`.
2. **Lives:** as each birth finishes, its two lives are launched by a detached per-seed chain, whether or not the birth meets P16F7.1:

```
E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
E16_STREAM=drift python3 exp16_integrated.py D <seed>
```

3. **Tables:**
   - `E16_SEEDS=1100009,1200007,1300021,1400017,1500007,1600033 E16_SUMMARY=SUMMARY_E16F7.md python3 summarize_exp16.py`;
   - the hard pairs from `e16f7_gate.py`, a copy of `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/gate_by_item_type.py` with the six seeds appended and `REF` defaulting to `HEAD`. Nothing else changed; the diff is two lines and a docstring note;
   - S, the eighteen-seed list and the correlation from `e16f7_signature.py`, adapted from E16-F6's read-only script.

Logs are written to the session's scratch folder and copied to `exp45/logs/e16f7_<arm>_<seed>.log` and `e16f7_birth_<seed>.log`.

## Predictions (from the brief; not to be changed)

Define **S** = the learner's lowest mod-97 family mean at k = 12 over checkpoints 2, 3 and 4.

- **P16F7.1 (births)** Each birth reaches at least 85% at k = 12 on all four development operations (mod 89).
- **P16F7.2 (twin)** The drift twin ends at or above 90 / 85 / 90 on the three family means in every seed.
- **P16F7.3 (rate)** At least four of the six seeds end at or above 0.80 of their twin on all three families.
- **P16F7.4 (the signature, pre-registered this time)** Both clauses must hold:
  - every seed with S below 65 ends below 0.80 of its twin on held compositions, and every seed with S at or above 70 ends at or above 0.80;
  - over the eighteen drift seeds (the twelve on file and these six), the Spearman rank correlation between S and the final `+-` pair by final answer is at least 0.35.
- **P16F7.5 (hard pairs)** At most two of the six seeds reach 0.80 of the twin on the `+-` and `-+` pairs alone (final answer), while at least four pass the family gate: the family gate and the hard-pair gate keep disagreeing.

**Reading, written in the brief before the result:**
- If P16F7.4 holds, S is an early-warning measure worth carrying into later designs.
- If the correlation clause misses while the threshold clauses hold, S separates failures but does not rank the passing seeds, which is what E16-F6 saw.
- If a seed with S above 70 fails, the failure has no early signature of this kind.
- P16F7.5 is the hard-pair count on new seeds. It changes nothing about the gate as pre-registered, which stays reported as such.

**State of the twelve seeds on file, recorded before launch so the new seeds' contribution is plain** (from `e16f7_signature.py` and `e16f7_gate.py` at this commit):
- **The signature on twelve seeds:** S ranges from 47.2 (600013) to 87.4 (200003).
  - The two seeds with S below 65 (600013, 400009) are the two composition failures.
  - Two seeds lie in the 65–70 band with no clause (1000003 at 66.7, 800011 at 69.3).
  - The Spearman correlation between S and the final `+-` over the twelve is 0.399. S was chosen after seeing these twelve, so the clauses are already met on them by construction. Only the six new seeds test it.
- **The hard pairs on twelve seeds:** on the twelve drift lives, 3 reach 0.80 of the twin on `+-` and `-+` alone and 10 pass the family gate.

## Dry run

The two commands were run once on seed 500011 with `E11_STAGE_UPDATES=4`, at the same time, at 03:07 UTC. Each ran its seven stages without error, in 5 min 43–47 s of wall time:

| Life | CPU-s |
|---|---:|
| Learner | 257 |
| Twin | 266 |

The receipts and logs are in `exp45/archive_e16f7_dry/`. The learner's dry receipt is identical, score for score, to E16-F6's dry receipt on this seed (`exp45/archive_e16f6_dry/`) and to run B's (`E16F6_RUN_B/exp45/archive_e16f6_dry/`).

## Run identity

| Item | Value |
|---|---|
| Machine | `Linux vm 6.18.44-fc-v338 #1 SMP @0 x86_64 GNU/Linux`; 4 cores; Intel Xeon at 2.10 GHz |
| Software | Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6, one thread per process |
| Container | a fresh one, not E16-F6's: that was a 2.30 GHz Xeon with the same software |
| Harness | e78aa83e |

## Forecast

**Pace.** The dry run here cost 1.29× E16-F6's dry run (206 / 199 CPU-s). It matches run B's (249 / 257 CPU-s), which ran on a 2.10 GHz Xeon. Run B's full lives cost only 1.04× E16-F6's, because training scales differently from the evaluation that dominates a dry run. So the forecast takes run B's measured costs, plus 3% for this dry run's margin over run B's:

| Item | CPU-s |
|---|---:|
| Birth | about 3,050 (run B: 2,904–2,983) |
| Learner life | about 5,150 (run B: 4,955–5,119) |
| Twin life | about 4,500 (run B: 4,358–4,432) |

**Phase 1:** six births on four cores, about 1.3 h.

**Phase 2:** twelve lives, about 57,900 CPU-s, about 4.0 h on four cores.

**Expected:**
- about 76,000 CPU-s in all;
- about 5.3 h of wall time;
- the last life finishing about 08:45 UTC.

The brief's estimate is about 5.5 h.

## If the container restarts

A restart kills every process, detached or not. A life or birth killed before it writes its last record has not finished. Such a run will be launched again from its birth (or from scratch for a birth) with the same command and seed, its partial receipt kept as `*_killed.json`, and the report will say so. No finished life is rerun.

## Push target

This session can push to `codex/minifly-consolidated-20261023`, so it commits there, pulls with `--rebase` before each push and never forces.
