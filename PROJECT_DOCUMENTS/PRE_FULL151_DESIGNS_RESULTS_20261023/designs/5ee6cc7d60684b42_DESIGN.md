# FE0 retention under the causal byte protocol — design (fixed before any run)

**Date:** 25 September 2026. Its SHA-256 is recorded in `RESULTS.json`.
**Question:** under the current `CAUSAL_DT` protocol, does FE0 still hold an old association after 24 hours, the re-teaching of neighbouring pairs, and another 24 hours?
**Status:** a bounded engineering experiment. It is not a full-life confirmation, and the S1–S5 roadmap and S2 are unchanged.

## Worlds and schedule

- **Worlds:** 16 new worlds, namespace `BYTE-LEARNER-FE0-RET`, block `RET`, indices 0–15. R2, massed, frozen BYTE-CORE-1 generator.
- **Schedule:** each world follows its own event list up to the end of challenge 0:
  1. first block: 48 pairs × 6 bouts;
  2. 24-hour rest;
  3. challenge 0: 12 of the old pairs for 12 bouts each (4 stable, 4 revision, 4 anomaly), with no new pairs;
  4. 24-hour rest.
- **Driver:** the `CausalFly` adapter and `present()` from `../BYTE_LEARNER_V8_REFINED_20260925/v8_trial.py`, unchanged; its hash is recorded.
  - Records are 14 bytes, with bytes Δ = 30/14 s apart and record starts 165 s apart.
  - `Teach` follows each outcome byte, and each interval is committed once.
  - A rest event advances both clocks by 86,400 s.

## Arms

- **Front end:** FE0 in both arms.
- **Inputs:** both arms get identical bytes, times and `Teach` events.
  - **A (early write):** `plastic=True` throughout.
  - **B (early no-write):** `plastic=False` during the first block only, then `plastic=True` from challenge 0 onward.
- **What a difference means:** a difference between A and B can come only from the first block's writes and their downstream effects.

## Readout

**The clean paired probe.** For each pair, each cue is presented to a separate copy of the current learner: 12 context bytes from time t, with the wrapper given by the cue's presentation count mod 2. Each copy is read at t + 12Δ, with no outcome and no teaching. The margin is v(neutral cue) − v(aversive cue), using the pair's original truth. A choice is correct if the margin is > 0, and counts 0.5 if the margin is 0.

- **Pairs probed:** the **36 first-block pairs that challenge 0 does not touch**. They are identified from the world's evaluator metadata, which the learner never sees.
- **Times of reading**, each at the world clock of the corresponding scheduled probe:
  - **T1:** immediately after the first block;
  - **T2:** after the first 24-hour rest, before challenge 0;
  - **T3:** after challenge 0 and the second 24-hour rest.
- **Per-world values:** the choice rate over the 36 pairs and the mean margin, for each arm and time. Dₜ is A minus B.

## Decisions

Statuses use 95% t-intervals over the 16 per-world values: POS if the lower bound is > 0, NEG if the upper bound is < 0, UNRESOLVED otherwise. UNRESOLVED is never read as "no difference".

1. **Gate: D1 choice.** An immediate difference exists if D1 is POS.
2. **Primary: D3 choice.**
   - D3 POS with the gate passed means an FE0 byte-cue association survives 24 hours, the re-teaching of neighbouring pairs, and another 24 hours under this protocol. This is an engineering screen, not a confirmation.
3. **Locating a loss.** If D3 is not POS, D2 shows whether the difference was already gone after the first 24 hours (decay) or disappeared only after challenge 0 (interference). D2 is secondary and descriptive.
4. **Always reported:**
   - the margins D1–D3;
   - each arm's absolute choice rates;
   - a specificity check at T3 on the world's **24 never-trained pairs**. Above-chance values there would indicate generalization rather than cue-specific retention.

## Checks, each with a control

- **Accounting.** Per record: 13 byte commits, 1 teaching event and 1 gap. Each rest event adds 86,400 s. The brain, sensory and world clocks agree.
- **Identical codes.** Both arms get identical pre-outcome codes on every record. PC: two different cues' codes differ.
- **B's first block does not write.** B's fast and slow memory are exactly zero at T1. PC: A's are non-zero there, and B's become non-zero after challenge 0.
- **Probes are pure.** The learner's digest is unchanged across every probe set. NC: a probe that commits on the real learner changes the digest.
- **Probe set.** The probe set is exactly 36 pairs, disjoint from challenge 0's 12 targets. PC: all 12 targets appear in challenge-0 trials.

## Not changed

- the teaching dose `event(30/14)`;
- the readout family;
- the text head.
