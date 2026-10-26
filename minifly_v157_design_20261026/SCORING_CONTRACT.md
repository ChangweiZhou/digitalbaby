# V157 common scoring and evidence contract

This schema reconciles lanes; it does not rewrite historical contracts or merge incompatible scores. The standing parent MINIFLY_OBJECTIVE_CONTRACT remains authoritative. Taught-cue retention is E1-R, not E2. E2 requires a prespecified never-reinforced label-preserving presentation change. E3 requires never-taught relations and relevant prior-teaching/shortcut controls.

## Answer path

The candidate receives raw prompt bytes and generic record boundaries, never an operand parser, expected answer prefix, answer length or digit mask. Every payload byte 0–255 is valid; EOS is distinct class 256. Generation selects each next byte itself and ends with its own EOS; a budget cap is CAP. Candidate parameters, stores and clocks are immutable during scored generation. Correct means the whole payload and EOS match. First-byte correctness, teacher-prefix loss and margins remain separate columns.

Expected outputs may be real feedback during warmup. Predict before revealing each actual target. Generated guesses are never relabeled as external observations. Scalar-only feedback is a different information budget and is not tested by A1. Arithmetic labels or synthetic task rules can live in the external data generator/scorer, never the candidate representation or control flow.

## Append-only evidence row

Record at least:

- **Identity:** version/lane/stage, evidence level, status, parent/architecture/update IDs, native/candidate/reference/witness role, source commit/hash closure, original receipt.
- **Actual operation:** encoder, cue path, memory, readout, mutable state, update equation, warmup equation and actual counter, context/own-output caps, EOS/tie convention.
- **Data/exposure:** dataset/split hashes, world, initialization and order seeds, exact/semantic held membership, unique cues, physical presentations, repeats, prior checkpoints and inherited shared prefixes.
- **Feedback:** external expected payload bytes, EOS and scalar verifications separately; writes enabled/inhibited, target arrival timing, self-generated events, replay and optimizer label reuse.
- **Controls:** prior-teaching inhibition versus later no-write controls, clocks/observations, recitation/frequency counters and the precise scope of positive controls.
- **Counts/results:** complete own outputs with denominators; OLD/NEW/held separate; target-level diagnostics; independent-world versus correlated-branch counts; first response before feedback.
- **Resources:** summed CPU including children/failed attempts, wall, peak RSS, fixed/mutable/temporary bytes, actual visited edges/neurons, global scans/sorts, cloning and minimum copied bytes; disclose missing instrumentation and unmetered overhead.
- **Decision:** prespecified caps/stops, accepted/failed/unresolved status, technical versus behavioral qualification, integration/adoption authority and next action actually authorized.

Use source reports to bootstrap the registry; do not rerun frozen experiments. Distinguish `NOT_MEASURED`, `UNRESOLVED` and zero. Text BPS cannot be subtracted from addition accuracy, and free-readout/reference gains do not count as local native gains. A reference that fails qualification leaves that instrument unresolved; it does not veto another lane.

## Reporting and promotion

Use complete native addition per total feedback and compute as the final comparison. Keep acquired-subset retention conditional and report its denominator. More cues or repeated queries do not add independent lives. An exact-key table can pass E1; that does not invalidate an E1 diagnostic, but forbids calling it transferable arithmetic.

A learned rule can be genuine E3 when the causal state comparison supports it. An exposure-independent shortcut, supplied arithmetic algorithm or externally optimized decoder cannot be credited to the native learner. Promotion requires fresh locked replication with delayed/intervening-learning probes and an explicitly authorized adoption decision. No single development result, finite interpolation or mechanism name suffices.
