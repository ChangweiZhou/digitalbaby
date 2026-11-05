# Round 28 small replication — partial checkpoint recovery

The supplied export enabled a small executed replication. Completed 32 evaluations: eight fresh lifetimes under three arms (24), plus four original validation lives under the initial and generation-10 genomes (8). No new evolution search was run.

## Scope

Recovered the original generator, learner and fast-write ablation through literal source edits. The latest complete printed genome is Run A generation 10, rounded to three decimals. Final generation-40 genomes for A/B/C are truncated in the export. This is therefore a partial-checkpoint replication, not confirmation of Round 28's final scores or all three runs.

Fresh seeds were fixed to 280101–280108. Each life contains two history worlds and one fresh world with 30 inline questions. Network birth and world RNG follow the recovered runner's same seed convention. Controls share identical byte histories and network births. Fast-off disables both recurrent and fast-readout writes from birth; the slow readout still learns at every byte. This differs from freezing already acquired fast weights at a later checkpoint.

## Fresh-lifetime results

Equal-weight means over eight lifetimes. Lower scores are better. Answer spans in the original generator include the trailing newline.

| Genome / fast writes | Fresh-world bpb | Bits per answer including newline |
|---|---:|---:|
| Initial / enabled | 4.069 | 22.262 |
| Generation 10 / enabled | 1.283 | 10.601 |
| Generation 10 / disabled | 1.230 | 10.225 |

Disabling fast writes improves bpb by 0.053 (paired seed bootstrap 95% interval 0.046–0.063) and answer cost by 0.376 bits (0.172–0.680). These descriptive intervals concern eight world/birth seeds conditional on one recovered genome. They do not establish repeatability across independent searches.

Search's recovered checkpoint predicts substantially better than the initial genome. At generation 10, fast writing harms the measured predictions slightly. This supports examining the slow-readout explanation and provides no justification here for a larger fast-memory search. It does not prove that generic recurrent states cannot bind entities, or determine what the missing final genomes would do.

## Original validation parity

Recovered initial genome: 3.98448 bpb / 24.5841 answer bits. Export: 3.98499 / 24.6086.
Recovered generation 10: 1.34812 bpb / 10.7154 answer bits. Export: 1.34774 / 10.7119.

Scores are close, not bitwise identical. The exported generation-10 genome is rounded, and numerical-library/runtime details of the original are unavailable. Native fast-enabled instrument matches the recovered original implementation exactly in the current runtime. All-writes-disabled produces 8 bpb as expected from the uniform byte head.

## Limits and unrun work

- Original scoring uses a 1e-6 probability floor. Scores preserve that convention rather than replacing it with exact unclipped log loss. The candidate softmax itself is normalized and predicts before byte reveal.
- No generation-40 genomes, B/C evaluation, fresh ES restart, equal-budget random search, new GLN run, N=1024 scale test, post-history freeze fork or autonomous answer generation was run.
- No full biological size or deployed-memory admission is claimed. The recovered model uses dense arrays for its fast matrices despite sparse random fixed connectivity and k-WTA activations.
- Retained outcomes are scalar per-life metrics, source and stream hashes, rather than per-byte traces or trained-state checkpoints.
- Worker training CPU totals about 25.8 seconds; run wall time about 13.4 seconds with two single-thread workers. These exclude source recovery and technical checks. Per-worker RSS receipts are not a measured process-tree peak.

Next useful input is the original final JSONL logs containing full genomes. Saved-genome evaluations can then test the memo's main negative result cheaply, before another search or an addressing extension.

## Files and reproduction

`run.py` executes all 32 declared evaluations and writes PLAN.json, RESULTS.json and SUMMARY.json. `check.py` verifies original-source parity, determinism, fresh seeds and the uniform no-write reference. NumPy 2.4.0 was used. Set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1 before launching Python. `recover.py` requires the original export at its recorded attachment location; normal evaluation uses the included recovered files and does not require it. source/ files are provenance data, not shell scripts to execute.
