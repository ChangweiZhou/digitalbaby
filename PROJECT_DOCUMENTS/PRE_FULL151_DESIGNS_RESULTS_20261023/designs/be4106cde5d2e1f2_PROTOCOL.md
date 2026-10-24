# V33 — Position-Bound Byte Memory Isolation

Locked before V33 scientific outcomes. The user's requested scientific roster is exactly 24 trajectories: V32 Stage B seeds 33101–33108, each with lag/ordinary (L-O), lag/full covariance (L-P), and the unchanged exact-context count dictionary (L-Dict). No conditional stages or extra scientific arms. These are paired interventions on known V32 seeds, not independent confirmation. Engineering checks use different seeds and never select parameters.

## Exact intervention

Reuse V32's lag encoder without redesign: the last 12 raw bytes, in temporal order, each represented by its eight signed binary bits, concatenated into 96 coordinates and normalized. Missing initial lags are zero-filled. Full-window byte identity and position are recoverable. The router retains exactly V32's seeded eight normalized 96-D centres and argmax rule. The representation changes the router's input, not the router rule. This isolates the complete encoding interface; it does not separate positional binding from changes in bit encoding, feature scaling, or induced bank allocation.

The V32 stream generator, RNG calls, byte order, schedules, markers, separators, noise, revisions and probes are reused verbatim. Neural writes retain float64, 0.6 current-score correction, temperature 0.1, eight 96×256 weight banks, P initially I, and V32's orthogonal-component cap ||q||/||q0||<=1/0.6. There is no RLS weight rule, finite horizon, gate, bank splitting, new encoder training, or covariance approximation. The dictionary retains exact last-12-byte categorical counts with a uniform total-one Dirichlet prior and no suffix backoff.

The autonomous interface remains past bytes → prediction → observed byte → one update. No evaluator metadata enters encoder, router or learner. No live state is reset at a record or wave. No replay of untouched A is allowed. Future-key representation diagnostics are evaluator-only and cannot affect learning or routing.

## Workload and counts

Exactly V32's A→B→C→D sequence: 16 six-byte keys per wave, 32 initial target presentations per key, alternating two learned renderings; four A keys corrupted once each after B; four disjoint A keys revised eight times each in rendering 0 after C. D supplies further unrelated interference. Eight untouched A keys never return after A. A third syntax is only probed, never learned.

Each trajectory has 64 unique logical keys, 128 distinct learned target contexts (two renderings each), 64 additional never-trained alias target contexts, 2,084 target prediction events, and 29,176 total next-byte prediction events. Revised/corrupted targets do not create new past-byte target contexts. Report separately the actual number of distinct raw 12-byte windows across *all* byte positions, distinct context/target pairs, per-key and per-rendering presentations, and 32 nominal initial presentations per mapping. Repetition and arm reuse are not additional unique examples. The eight underlying streams are each used by three writers.

Historical current-code ordinary/full and dictionary results are embedded as compact immutable reference records, with V32 source, archive, job, and stream hashes. Every V33 stream must match its V32 hash. The dictionary is rerun as one of the requested 24 trajectories and must reproduce historical endpoints; the current-code neural trajectories are not rerun.

## Primary endpoint and prespecified decisions

Untouched-A accuracy after D, averaged across both trained renderings, is primary. Always display A after A beside A after D. The independent unit is a seed. All contrasts use complete paired two-sided 95% Student-t intervals; missing seeds do not pass. Secondary intervals are descriptive, without multiplicity adjustment.

**Independent covariance contribution** requires jointly:

- L-P and L-O both have mean initial untouched-A accuracy >=0.50 and its 95% lower bound >0.25 (operational acquisition gate).
- Mean L-P minus L-O untouched-A after-D accuracy >=0.10, with paired 95% lower bound >0.
- D current accuracy noninferiority: paired 95% lower bound for L-P minus L-O >=−0.05. `D_current` is the final after-D accuracy on D's own 16 keys and both trained renderings, not a stream-wide average. C-current and early 1/2/4-presentation learning are also reported, without silently replacing this gate.
- Target-only prequential loss noninferiority: paired 95% upper bound for L-P minus L-O <=+0.05 bits/target byte, over the complete unchanged stream.

The 95% bounds make the user's noninferiority tolerances explicit. **High-retention covariance candidate** additionally requires L-P mean absolute untouched-A accuracy after D >=0.75, retaining the prior round's descriptive high-retention level. A contribution claim can pass without this level; the two claims must not be conflated. Neither establishes fast alias revision or a general persistent core. No confirmation automatically follows success.

Normalized accuracy retention is (A_afterD−chance)/(A_afterA−chance), only for seeds with A_afterA>=0.50. Use 1/256 for the unchanged untrained-output baseline and also report a labeled 1/24 target-support chance sensitivity. For log loss, use (8−H_afterD)/(8−H_afterA) only when acquired gain>0.5 bits and accuracy>=0.50. Do not clip fractions or suppress negative values. Keep raw accuracies/losses alongside every fraction. Gain against uniform includes alphabet learning, so it is not a pure fraction of identities stored.

Paired representation effects compare L-O/current ordinary and L-P/current full on the same seeds. Difference-in-differences of after-D covariance advantages is also reported. These are not fresh confirmation, and their causal scope is the full representation intervention.

## Alias, revision and noise diagnostics

Before training, encode all 64 keys in all three renderings. For forms (0,1) and (0,2), report per-key cosines and bank equality, the same-key mean, and the corresponding different-key cross-form mean over all ordered i≠j. Identity separation is the difference between those means. Report same/different-key bank-agreement rates, unique codes, and per-alias bank occupancies. A positive gap alone does not certify adequate invariance. The same read-only calculations for V32's current code contextualize the historical intervention; they add no learning trajectory.

Revision probes at 1/2/4/8 changed labels separate exact rendering, previously trained alternative, and never-trained alias, including target and obsolete-target probabilities. Predictions before each revised target are aggregated by exposure. Early acquisition aggregates distinguish second logical-key occurrence (new rendering) from second occurrence of an exact rendering. Noise damage uses pre/post-corrupt-record probes on the corrupted key and unrelated untouched A.

## Covariance and private-lag diagnostics

At every after-wave checkpoint, record each bank's effective rank of C and of data geometry C−I, trace(P), eigenvalue range, write occupancy, and observed all-byte/target-byte mean z′Pz, capped and uncapped ||q||/||q0||, cap rate, and Frobenius write norm. Record target-context bank occupancy separately from occupancy over all bytes. P and W are never changed by measurement.

Since a write is ΔW=0.6 q residual′, its Frobenius norm is computed exactly as 0.6||q||||residual||. After A acquisition, the fixed untouched-A probes start an evaluator-only disturbance ledger. For each write, compute the root-mean-square score change over the fixed old probes and 256 outputs, including zero effects in other banks; accumulate the sum of these RMS step changes and the sum of their squares. Track trained and held-out A probes separately. At each wave report the net RMS score difference from the after-A snapshot. Cumulative variation is not a signed forgetting attribution and includes helpful changes. No per-byte disturbance dump is retained.

For lag blocks, save target-code squared norm per lag and centered between-key/alias variance per lag. Full windows have 1/12 norm per block by construction; variation, not equal norm alone, carries information. Reconstruct C from P only in the evaluator and save each 12×12 matrix of ||C_lm||_F and ||(C−I)_lm||_F. This does not introduce block covariance learning. Persist only aggregates/per-key static diagnostics, not matrices of model weights or per-presentation logs.

## Interpretation boundaries

High L-O retention with a confidently small L-P increment favors simplifying memory for this workload. A large, noninferior L-P gain with high absolute retention supports fresh confirmation before compression. If both fail despite improved identity geometry, the existing capped shared-bank implementation remains inadequate; this is not an impossibility theorem for all shared parametric memory. Good exact revision with weak alternative/held-out revision leaves invariance unresolved. High cap usage makes the current cap/direction construction a suspect, not a proven cause. RLS, delayed contacts, temporal permutations, episodic slots, and private/block covariance are reserved for a later user-authorized round.

## Execution contract

One self-contained .py; one command; NumPy/SciPy; automatic macOS caffeinate; one BLAS thread; atomic per-job results and archive; fixed source/plan/reference manifests; a process lock; strict resume validation. Outputs: REPORT.json, SUMMARY.csv, representation/alias diagnostics, covariance/cap diagnostics, revision curves, costs, source, protocol, compact historical references and manifest. Exactly 24 scientific jobs, even on success. No outcome-dependent roster expansion.
