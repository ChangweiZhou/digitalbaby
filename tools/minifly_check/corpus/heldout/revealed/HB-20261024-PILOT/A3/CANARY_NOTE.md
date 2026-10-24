# Canary A3 (DIRTY) -- held answers leaked through a declared non-feedback stream

Attack layer: learner / measurement. The lock declares a second exposure stream, `manifests/CAL/*/*.jsonl`, with **feedback_bearing: false, phase "calibration"** -- so it looks like an authorised, benign warm-up channel. Its rows are the **HELD** items (prompt + correct answer), committed with the cycle-0 streams. The candidate W is thus shown the held answers through a channel the delivery calls non-feedback, while the TAUGHT manifest that the scorer uses for strata stays clean, so the held items still classify S2/S3 determined and are counted. W then "generalises" (the base PASS numbers) because it was fed the answers.

## Why it violates the spec
- SPEC I3 / 8 exposure_streams: held information must not reach the learner through any exposure stream. SPEC FM-12 names exactly this route -- "held information reaching the learner through ... calibration ... while the exposure audit checks the manifest rather than what the learner was fed." ADM-EXPO is the "canonical intersection plus undeclared-stream rejection": the union of ALL exposure streams (feedback-bearing or not) intersected with the held set is non-empty -> leakage.
- A strict reading of SPEC 9.3 also makes those held orbits "taught" (they occur in an exposure stream), so they become S1 and are excluded from E3 -- either way the PASS is illegitimate.

FM class: FM-12 (exposure leakage / illegitimate task-shaped features).

Detection: ADM-EXPO at the roster and official gates. The stream is DECLARED, so the undeclared-file branch does not fire; the content-intersection branch must. A checker that only audits feedback_bearing streams (overfit to the visible FM-12 fixture) would miss it -- that is the generalisation this canary probes; a faithful ADM-EXPO per FM-12 catches it.

Consistency: the only differences from the base are the extra exposure_streams entry (cascaded into every lock-sha consumer by the builder) and the CAL manifest files. W's outputs, receipts, G and headline are the base PASS values -- the delivery looks like a clean generalising learner, which is the point.
NOT a literal copy of C12: new stream name/phase ("calibration", non-feedback) and layout.
