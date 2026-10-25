# V152 — three-cycle exercise

**Verdict: REFERENCE_NOT_READY.** The bounded exact-BPTT reference did not acquire complete taught answers, so the design stopped before fresh assessment. This is an accepted negative assay, not a native learning improvement or a proof against local learning.

The goal remains retained, complete own-byte/EOS unseen-addition generalization per total feedback and compute. Gradual warmup, raw bytes and unrestricted 257-class output were retained; no arithmetic feature, rule, forced stop, attention, Transformer or ELM was added.

## What ran

- Six fixed calibration groups: R_BP, R_LE and R_LE0 at η=0.01 and 0.03, on the one predeclared calibration world.
- Each group completed W/SH_OLD/N_OLD, sharing the learned OLD prefix physically. These are 18 correlated histories, not 18 independent worlds.
- 15,360 physical record applications and 43,344 revealed reply/EOS targets, including inhibited writes. Only 128 unique taught relations were used repeatedly.
- Technical work adds 53 records / 106 targets. Total: **15,413 record applications / 43,450 targets**. There were 24,837 own/teacher-forced query pairs, including 69 technical pairs.
- The three fresh assessment worlds, native efficacy comparison, final native bias ablations and assessment-state credit diagnostics were **not admitted**. Native adapters received technical qualification only.

## Results

The shared rate selector chose η=0.03 from the equal-method final-W validation loss. Selection used 32 validation relations; it did not use the unseen assessment set.

| Reference | OLD exact after OLD | NEW exact after NEW | Immediate exact, all updating branches | Final W OLD+NEW BPS |
|---|---:|---:|---:|---:|
| R_BP | 0/64 | 0/64 | 0/1536 | 3.1981 |
| R_LE0 | 0/64 | 0/64 | 0/1536 | 3.1248 |
| R_LE | 0/64 | 0/64 | 0/1536 | 3.2031 |

Across both rates and all methods, immediate complete-answer acquisition was **0/9216**. By OLD pass 8 every W query emitted EOS before producing any payload. At final NEW, all 128 W queries per method/rate did the same. Teacher-forced EOS argmax accuracy was 100%, while first-answer-byte argmax accuracy was 0% at these checkpoints.

For R_BP at η=0.03, OLD loss fell from 8.0156 to 3.6485 bits per target, and final combined OLD/NEW loss was 3.1981. This loss reduction did not produce an answer. R_LE0 even achieved slightly lower final combined loss than R_BP (3.1248 versus 3.1981), while both answered zero problems; this is no evidence that removing temporal credit helps arithmetic.

## What this diagnoses

The directly observed failure is **premature termination and failure to acquire complete taught sequences**, already present before interference. It cannot be described as forgetting alone. Exact BPTT and both local methods share this failure, so this run does not isolate missing cross-neuron temporal credit as its cause.

The learned class bias is a plausible contributor to a frequency-like EOS policy, but no reference-bias causal ablation was authorized or performed here. Input-state scale, the fixed C16 port, optimization dose and bias/core competition remain unresolved. We do not claim that a larger decoder, more passes, a new topology, or arithmetic features would fix it.

The failed comparator means the intended native-versus-reference bridge is not qualified. It says nothing new about whether the three native candidate arms improve unseen addition: they were not assessed. The assessment targets remain unscored.

## Three cycles and qualification history

1. **Design / audit / interface test / revise:** checked exact output gradients, exact analytic BPTT with coordinate finite differences, forward parity, local eligibility in the diagonal-time limit, all raw payload bytes, read-only generation, no-write counters, and zero-error native A parity with the operative parent. A bytewise-pickle resume check falsely failed even though all arrays and counters were bit-identical. Its source and failed output were preserved; only the checker was revised.
2. **History audit / bounded test / revise:** a duplicate metadata-key error stopped calibration startup before its first training record. That attempt, its trace and its 64-query count were preserved; the discarded baseline prediction rows were not retained. The queries were paid from the technical-query reserve. The logging schema was corrected; no trained trajectory was restarted. All six fixed calibration groups then completed. The shared-rate readiness gate failed; no wider search was added.
3. **Saved-output audit / response reduction / revise:** a separate reducer verified 65 source hashes, all six receipts, every recorded own-byte/EOS answer and teacher-forced loss, physical history/exposure counts, warmup counters and the failed gate. No learning trajectory or held assessment was rerun. The revised design narrows the next causal question to generic output-bias versus context learning.

The first 15-coordinate derivative assertions passed before the initial checker failure, but their intermediate numeric residual array was not saved. The supplementary paid check retained residuals; its largest coordinate error was 5.56×10⁻¹¹. Native parent state parity error was exactly zero. The qualification record explicitly preserves this evidence limitation.

## Cost, exposure and evidence boundary

The main ledger reports 26.58 accounted CPU seconds, including a 1.6-second conservative reserve for a separate no-feedback serialization inspection; peak recorded RSS was 175.3 MiB. Execution wall accounting includes the earlier preflight/startup interruptions. Postrun saved-response audit and packaging cost are reported separately in POSTRUN_COST.json. Numerical execution used one worker and one numerical thread. These are small diagnostic costs, not a native efficiency or scaling result.

Queries additionally evaluated 69,918 true reply/EOS positions without learner writes. Those are distinct from the 43,450 training/technical target applications. Validation labels were an information source for rate selection and are disclosed separately; held assessment labels were not scored. Exact and canonicalized lookup could recite taught facts; favorite-reply and position-frequency controls were specified for E3 but not reached. No E3 or broader capability claim is made.

E0: technical interfaces/derivatives qualified after documented checker/startup corrections. E1: acquisition was tested and failed the complete-answer readiness criterion. E3: not run. No core was adopted, no arithmetic pointer was replaced, and nothing was pushed to GitHub.

See [revised design](REVISED_DESIGN.md), [independent audit](FINAL_AUDIT.json), [analysis](ANALYSIS.json), [counts](COUNTS.json) and [source lock](SOURCE_LOCK.json). Raw checkpoints and detailed per-query datasets remain local.
