# V6-R1-R2 prospective revision — correct the generic sensory signal scale

Proposal only, not a launched run or adopted core. Read the completed `REPORT.md` and `REVISION.md` alongside this document. Preserve V6-R1 as the comparison. Do not import an external decoder, attention, Transformer, ELM, arithmetic features, carry units, answer lengths or worked solutions.

## Why this intervention

R1 initializes every nonbias column with SD `0.05/sqrt(348)`. That assumes roughly 348 active presynaptic contributions. Its byte input is one-hot: exactly one of 259 symbol columns is active. At zero previous state, content-mean RMS per neuron is therefore only about 0.00268, versus proposal noise SD0.1. The ratio is about 1:37. This is a generic initialization/statistics issue, not a reason to hardcode number representations. The actual OLD-prompt address diagnostic in `REVISION.md` tests its practical relevance; it does not establish a causal cure.

A mathematical score identity can be correct while its feedback has too little signal to learn useful representations. Conversely, EOS failure after wrong generated prefixes does not alone prove a defective EOS codec. Do not replace either issue with an analytic stopping rule.

## One prospective core intervention

Keep H88 and the same stochastic gated recurrent equations, local score eligibility, record-end updates, native CANON circuit and native output signs. Retain sigma0.1, gate bias log(0.1/0.9), both gradual schedules, calibration, sample roster sizes and all teacher/probe ordering.

Initialize the two input groups using their actual fan-in, for both Wv and Wg:

- One-hot symbol block: independent zero-mean Gaussian SD0.05. One active contribution now has the original intended RMS0.05.
- Recurrent block of 88 coordinates: independent zero-mean Gaussian SD`0.05/sqrt(88)`.
- Biases remain content0 and gate log(0.1/0.9).

This is one block-initialization intervention, not a grid over noise, rates, rewards or arithmetic curricula. Symbols remain exchangeable learned columns; digits, operators and protocol events receive no semantic vector or privileged coordinate. Full learning stays active in the candidate. The fixed-controller arm is only an ablation with the same native plastic circuit, not a separate random-feature answer solver.

This does not fix the high-variance learning rule by theorem. It is a falsifiable first correction to a concrete birth-scale mismatch. Do not silently replace stochastic training with mean-state evaluation or reduce the noise after seeing scores.

## Matched addition comparison

Use two fresh engineering worlds, locked before feedback. Do not count the already inspected R1 worlds as confirmation. Retain 0..9999, 128 OLD/32 NEW/64 globally held unordered orbits, all sixteen operand-length strata, balanced orientations, OLD32 passes and NEW8 passes, both rests, three evaluation tapes, and the existing checkpoint schedule.

Five arms per world:

1. ORIGINAL: unchanged R1 learned controller and native values.
2. RESCALED: the single block-initialization revision, learned controller/native values.
3. RESCALED_FROZEN: revised birth with controller fixed, native learning on.
4. RESCALED_NO_OLD: both learning paths and baseline off during OLD, both on during NEW, same clock.
5. RESCALED_SH_OLD: within-length deranged OLD labels, correct NEW, both paths on.

The original API cap guard may be shared by all arms because it is numerically inert for the bounded roster; lock and test it before feedback. Ten correlated histories in two independent worlds imply 43,520 external record applications. Lock exact event/feedback counts from the new manifests. Count every prompt byte, expected answer byte, EOS, repetition and generated/private view. There is no free byte pretraining, extra problem set or hidden warmup.

Keep the held-out exact own-output gates and causal guards from R1. Additionally compare RESCALED directly with ORIGINAL at common sample and measured compute budgets. Taught acquisition remains a diagnostic; a memorizing revision with no held advantage has not met the goal. Predeclare the observed-bit/EOS-prior countermodel alongside raw/canonical/frequency/length/suffix0–4 models. No target-length oracle may supply scores or outputs.

## Preflight, cost and decision

Before feedback: enforce generic caps and invalid-payload atomicity; verify all byte/EOS transport identities, active and closed Gaussian/gate score finite differences, causal baseline/gradient ordering, readonly generation, restoration and independent mutable states. Resolve actual imported asset paths explicitly in the source closure, not just identical vendor copies. Hash the complete reducer before any held probe.

Measure OLD-only prompt sensitivity and tape sensitivity at birth without using labels or held cases. These measurements diagnose the intended scale change; they are not a tuned efficacy threshold. Then run the full bounded addition comparison. Do not substitute serial recall or a new task for it.

Proposed resource ceilings stay one worker/one numerical thread, 2GiB RSS with 2GiB admission headroom, 7,200 CPU seconds and 10,800 active-wall seconds, including qualification, all probes and packaging. Profile the first complete planned history with all probes and admit the remainder conservatively with audit reserve. If it does not fit, preserve the partial run rather than removing controls, shortening ranges or extending the ceiling. A future run needs explicit authorization; none is started by this revision.

If rescaling improves conditional input coding but exact taught answers still fail, the remaining candidates include native capacity/interference, readout conditioning and credit variance. Inspect those measured failures before choosing another operation. Do not automatically add the earlier population expansion, replay or a larger decoder. If taught acquisition succeeds but held causal gates fail, report that distinction and do not rename recitation as abstraction.
