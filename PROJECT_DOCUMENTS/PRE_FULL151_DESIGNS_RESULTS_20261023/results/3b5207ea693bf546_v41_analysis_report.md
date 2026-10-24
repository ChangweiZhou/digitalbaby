**V41: what the three mechanism experiments establish**

All 12 scheduled jobs finished and were recorded. Eight hybrid/stationarity jobs completed their full workloads; three utility jobs reported failed predeclared gates, and one executed its eviction comparison. No thresholds, seeds or workloads were changed after seeing scientific results.

My strongest recommendation is to retain the separation between retrieval and writing. READ-HYBRID combines immediate transfer with SLOTS’ durable history on this task. Feeding the stronger fallback into admission changes which memories survive and sacrifices some old recall. Freezing normalization helps, but the stationary lag representation remains the better reference. Utility-based eviction remains unresolved because the assay was valid in only one seed.

The latest recoverable chat recommendations were three small closure experiments: a common eviction event, a read-versus-writing hybrid split, and lag/live/frozen representations. The complete numerical protocol was unavailable. DESIGN.md therefore identifies our numerical choices explicitly as refinements, rather than claiming exact reproduction of an unseen design. These are new focused experiments, not directly comparable reruns of V40’s larger workloads.

| Experiment | Independent seeds | Training per seed | Status |
|---|---|---|---|
| Utility | 4 | 320 contextual next-byte observations | 1 valid; 3 gate failures |
| Hybrid | 4 | 15,344 continuous bytes / 1,096 records | 4 completed |
| Stationarity | 4 | 4,088 warmup + 15,344 main bytes | 4 completed |

Utility contexts contain twelve supplied bytes that are covariates, not twelve extra learning events. Its valid seed evaluates 224 frozen target predictions over 48 unique followup contexts. Repeated predictions, identities, aliases and checkpoints are not independent replications. Across all jobs, 139,104 continuous byte presentations plus 1,280 contextual target observations were used for training; the two kinds of samples should remain separately reported.

**The read-only hybrid preserves recall while improving one-shot transfer.** Means below are over the same four seeds. E1 is the held-out rendering probe after the first training presentation; old recall tests untouched first-wave identities after wave D.

| Arm | Whole bits/byte ↓ | Target bits/byte ↓ | E1 accuracy | Old held-out recall |
|---|---|---|---|---|
| BASE | 2.8329 | 5.1440 | 2.54% | 0.00% |
| SUPER | 2.3331 | 1.9586 | 100.00% | 70.31% |
| SLOTS | 2.5013 | 2.7674 | 2.54% | 96.88% |
| READ-HYBRID | 2.4396 | 1.7797 | 100.00% | 98.44% |
| FULL-HYBRID | 2.3936 | 1.7578 | 100.00% | 79.69% |

READ-HYBRID literally shares SLOTS’ memory and writes. Its whole-stream change is -0.0617 [-0.1268, +0.0034] bits/byte (paired mean and pointwise Student 95% interval). The target contribution is -0.0706 bits per whole-stream byte, while non-target bytes contribute +0.0089; they sum to -0.0617. Thus strong target transfer does not translate into an equally large whole-stream gain. Each record has one target byte and thirteen non-target bytes.

FULL-HYBRID minus READ-HYBRID changes whole loss by -0.0461 [-0.0805, -0.0117] bits/byte and old held-out recall by -18.75 percentage points. The old-recall interval is −43.11 to +5.61 percentage points, so this remains a four-seed descriptive warning. This comparison measures the total consequence of the different writing history; it does not attribute the loss to one particular admission event. FULL is not uniformly better simply because its average bits/byte is lower.

The independent event audit makes the durability concern more specific: FULL has zero consolidated-slot evictions in all four seeds, yet all thirteen of its old held-out failures lack memory authority. READ answers twelve of those thirteen correctly. FULL also has substantially fewer target promotions and unused slot capacity. This supports investigating suppressed consolidation; it does not prove an identity-level causal chain because trace eviction and failed matching remain possible.

READ still costs 0.10658 whole-stream bits/byte relative to SUPER. The authority-stratified audit localizes its non-target penalty to exact probationary traces and similarity reads from supported slots; fallback reads add zero relative to SUPER because they are identical there. This identifies the next read-authority question. Target/non-target evaluator labels must not become an oracle input to a future policy. See hybrid_scientific_audit.md for the full decomposition and seed values.

The hybrid uses the inherited 4,096-unit superposed layer: 8,388,608 output bytes plus 786,432 projection bytes, in addition to its slot memory and common learner. This is not a reduced-neuron or lower-memory result. Shared execution also cannot establish standalone runtime. Slot/traces stay within the fixed capacities of 384/512.

**Freezing normalization improves this phase model, but does not close the gap to LAG.** Both phase branches start with identical weights and normalizer state after causal warmup. Only normalizer updates stop in FROZEN; all 15,344 subsequent output updates continue in both branches.

| Arm | Main bits/byte ↓ | Warmup-inclusive bits/byte ↓ | E1 accuracy | Old held-out recall | Fixed-context active overlap |
|---|---|---|---|---|---|
| LAG | 2.3122 | 2.3766 | 99.80% | 70.31% | 1.0000 |
| PHASE-LIVE | 2.6591 | 2.9375 | 84.77% | 43.75% | 0.8378 |
| PHASE-FROZEN | 2.5521 | 2.8530 | 91.02% | 70.31% | 1.0000 |

FROZEN minus LIVE changes main loss by -0.1070 [-0.1549, -0.0590] bits/byte; FROZEN minus LAG is +0.2399 [+0.1423, +0.3376]. The discovered periods were [14, 28, 14, 14] bytes. Fixed-context overlap is measured at matched phase. Freezing removes normalizer drift there, but recurring contexts at their actual stream times can still receive different codes when period discovery selects a multiple of the fourteen-byte record length. Mean actual-time recurring-context overlap is 1.0000 for LAG, 0.9414 for LIVE and 0.9764 for FROZEN; FROZEN reaches only 0.9056 in the period-28 seed. The raw report includes both measurements.

Geometry changes and removed drift occur together in this intervention, so the results do not establish that drift alone mediates every prediction difference. All arms retain output plasticity and reach full current-wave acquisition in this run. Equal charged representation budgets include explicitly reported inert padding for LAG; that allowance does not improve its functional model.

**The utility comparison did not qualify as a four-seed experiment.** The fixed event occurred after 320 target observations in every seed, with no early eviction. Gates required at least sixteen resident slots, at least two postbirth utility observations for every resident, and at least 90% held-out acquisition.

| Seed | Resident slots | Held-out acquisition | Minimum postbirth uses | Outcome |
|---|---|---|---|---|
| 41101 | 42 | 95.83% | 1 | A resident has fewer than two postbirth uses |
| 41102 | 47 | 97.92% | 2 | Valid |
| 41103 | 48 | 100.00% | 1 | A resident has fewer than two postbirth uses |
| 41104 | 35 | 72.92% | 2 | Acquisition below 90% |

In the sole valid seed (41102), every policy demoted eleven of forty-seven resident slots at the same time. Utility-minus-LFU future loss was +0.071311 bits/target; utility-minus-random was -0.427865; utility tied shuffled utility to numerical precision. These are single-seed descriptions, with no confidence interval or general ranking.

The surviving utility fixture has a further limitation: base prediction loss is exactly eight bits on every held-out identity. It therefore supplies little independent variation in future predictive value beyond recurrence. It cannot establish an advantage of value over frequency. Demotion restores two exact traces: all five eviction arms use 152,988 semantic bytes versus 138,732 for no eviction. Equal victim counts isolate slot authority, but do not imply a RAM saving.

**What I would carry forward.** Keep READ-HYBRID’s firewall between fallback reads and corroborated writing as the reference composition. Keep LAG as the stationary representation reference. A future utility test should first make acquisition and postbirth exposure reliable across separate engineering seeds and vary base predictability independently of demand; then lock one new event and its controls. The present gates should remain visible as failed outcomes, rather than being weakened after inspection. A small phase-alignment test could address the residual time dependence before spending compute on another capacity sweep. Those followups were not run here.

All effect estimates are conditional on these synthetic workloads. The intervals are pointwise, exploratory, and based on four paired seeds; no multiplicity adjustment or broad qualification is claimed. v41_primary_contrasts.csv contains every seed difference and an exact sign-reversal sensitivity calculation. With four seeds the exact two-sided sign-reversal test has a minimum p-value of 0.125 and cannot attain p<0.05. A parametric Student test can produce smaller p-values under its distributional assumptions; those assumptions cannot be assessed reliably from four seeds. Utility has only one eligible comparison and is excluded from that calculation.

**Execution and reproducibility.** The unchanged runner embeds its source dependencies, saves complete state every thirty-two records, atomically replaces checksummed checkpoints, retains a previous generation, validates source/configuration identities, rejects concurrent writers, and skips verified completed jobs. SIGINT/SIGTERM stop after a block; rerunning the same command resumes. macOS caffeinate is automatic. Initial environment and invocation records are preserved. A checkpoint snapshot was saved during scientific execution. The initial three-worker run hit the environment memory limit after nine completions; its broken pool was interrupted and the three remaining jobs resumed sequentially from checkpoints with the same source and plan. The default is one worker. A fatal process-pool failure can require an interrupt before resuming; automatic recovery of a broken pool is not implemented. The final compact results ZIP omits completed checkpoint states.

Engineering checks include original numerical equivalence, read-only probe invariance, freeze-only-normalizer behavior, exact victim budgets, shuffled-score preservation, serialization/resume equivalence, and rejection of corrupted or mismatched records. The command-line smoke workflow paused, resumed all three jobs, and then reused all completed jobs with zero new learning. A final scientific rerun also verified and reused all twelve completed jobs with zero new learning. The numeric-key JSON checksum issue found during smoke testing was repaired before scientific execution.

Source SHA-256: `223434f801c3f51071b420bb7e60330795d8559d54bc24730c9bc424d77a4b27`. The analysis independently verifies all twelve result checksums, identity/plan links, denominators, read-loss decomposition, observed capacity bounds, normalization state, and validity-gate handling. See v41_integrity_audit.json, verification.json, and the execution logs for evidence.

Files: elm_v41_mechanism_closures.py runs or resumes the science and regenerates raw reports; analyze_v41_results.py regenerates this focused analysis and figure from the completed JSON records. v41_results_results.zip contains the raw records, plan, source, environment, per-seed metrics and paired summaries. The complete handoff archive also contains the protocol, verification evidence and focused reports.
