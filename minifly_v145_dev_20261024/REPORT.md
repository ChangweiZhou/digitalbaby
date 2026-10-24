# V145 three-cycle development exercise

**Status: three bounded technical cycles completed with limitations; main experiment not run.** No native efficacy, persistence, statistical confirmation or improved total efficiency is claimed.

## Results

| Code / downstream | Passes | Taught complete | Held complete |
| --- | ---: | ---: | ---: |
| SYMBOL / BP | 256 | 100.00% | 13.33% |
| ORACLE-LINE / BP | 256 | 38.75% | 26.67% |
| ORACLE-CIRCLE / BP | 256 | 82.50% | 16.67% |
| ORDER / BP | 64 | 72.50% | 6.67% |
| ORDER / DFA | 64 | 60.00% | 10.00% |
| PERM / BP | 64 | 57.50% | 10.00% |
| PERM / DFA | 64 | 30.00% | 0.00% |
| BAG / BP | 64 | 37.50% | 6.67% |
| BAG / DFA | 64 | 23.75% | 6.67% |
| NO-WRITE / BP | 64 | 50.00% | 0.00% |
| NO-WRITE / DFA | 64 | 33.75% | 3.33% |
| ORDER-BP / BP | 64 | 75.00% | 3.33% |
| ORDER-BP / DFA | 64 | 42.50% | 3.33% |

Each held percentage averages orientations within each of 15 withheld unordered orbits. Both orientations were excluded from feedback. Generation uses own bytes and own EOS within eight decisions. Supervised answer prefixes occur only in training. The three oracle/SYMBOL trials have 256 passes; the ten learned-code trials have 64 passes, so cross-dose differences are not mechanism effects. All trials use H128, rate .03, decay .001; pretraining uses d4, rate .01 and gradual 1,024-transition warmup. Addition retains the inherited gradual 1,024-record ramp.

Ordered counting does not beat the 13.33% COPY_A/COPY_B countermodel on this world. BP ties BAG and trails PERM; DFA improves over NO-WRITE but still trails copy. Counting can improve taught acquisition without demonstrating useful held generalization. Exact-gradient code pretraining also fails to improve held answers in this fixture. The number-line oracle is an evaluator-only arithmetic prior, never a candidate. Its 26.67% held result is not a reliable capacity ceiling: taught accuracy is only 38.75%.

## Three cycles and revisions

1. **Design/audit:** implemented the stated local and detached-target gradient updates. Finite differences agree to 1.03e-10. All 256 original receptor-byte traces match the inherited event API exactly. Zero initial learning rate and the fixed EOS target are checked. Generic code learner imports no task generator.
2. **Bounded test:** one fresh developmental world, three same-interface diagnostic trials and five pretraining conditions crossed with BP/DFA. Pretraining has equal token histograms and boundaries across ORDER/PERM/BAG/NO-WRITE. Original and final source hashes match. No main identity was generated.
3. **Repair/audit/revise:** a short fresh-world rehearsal reached checkpoint serialization and failed because a local FE class was not pickleable. All attempt artifacts were preserved. A separate fresh repair fixture uses a module-level front-end class; a live uninterrupted continuation matches the checkpoint-restored continuation. No completed Cycle 2 trajectory was replayed. Exact saved-output scores and roster were checked; missing, duplicate, false-score and NaN fixtures were rejected.

The original Cycle 1 pickle check tested only the small code predictor, not the combined addition/front-end object. This was insufficient qualification and exposed the Cycle 3 error. See REVISED_DESIGN.md for the repair.

## Exposure and cost

Total incurred: **113,681 record applications and 352,769 observed target tokens**, including NO-WRITE exposures, the failed checkpoint attempt, and the repair continuations. The four edge-update fixtures are listed as technical records, not addition examples. Counts include 10,240 Cycle 2 counting records, 128 failed-rehearsal counting records, all 13 Cycle 2 addition fits, the failed four-pass fresh-world fit and 225 repair records.

Measured CPU is a lower bound: 11.792 seconds through the last saved first-attempt progress marker plus 0.785 seconds for the repair audit. Startup/import CPU and the failed attempt's final interval were not captured. No all-in CPU-efficiency claim is made. Per-fit CPU was lost with the first process, so this is not a qualified campaign cost ledger.

## Limits and next decision

This is a shortened exercise, not the proposed full D4 protocol: no six-setting tuning menu, FROZEN-DOWN controls, BIRTH-MORE-ADD cost matching, four main worlds, 56-fit campaign or native integration was executed. Code geometry and per-condition nearest-neighbor controls were not retained. Final evidence is descriptive E1 acquisition and a negative/insufficient E3 readiness probe. Original-symbol nearest-neighbor is 0%; learned-code-specific nearest-neighbor performance remains unmeasured. No retention claim is possible.

Keep the arithmetic and own-output endpoint. Do not expand this into a main campaign on the present evidence. First qualify the corrected combined runtime and complete cost ledger; if further development is requested, diagnose the same-interface oracle acquisition and record predictive-code quality before choosing a single change. Nothing here licenses arithmetic features, attention, ELM, extra tasks or abandoning gradual warmup.
